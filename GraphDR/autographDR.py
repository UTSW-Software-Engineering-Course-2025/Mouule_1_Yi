from __future__ import annotations
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph,eye
from scipy.sparse.linalg import inv
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA

# -------------------------------------------------------------
# Models
# -------------------------------------------------------------
class DAE(nn.Module):
    """Simple denoising auto‑encoder."""

    def __init__(self, in_dim: int, latent_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, in_dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# -------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------

def load_dataset(limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    print("Loading Hochgerner 2018 dataset …")
    data = pd.read_csv("./hochgerner_2018.data.gz", sep="\t", index_col=0)
    anno = pd.read_csv("./hochgerner_2018.anno", sep="\t", header=None)[1].values

    # CPM normalisation (counts per million)
    per_cell_sum = data.sum(axis=0)  # (cells,)
    data_cpm = data / per_cell_sum.values[None, :] * np.median(per_cell_sum)

    # log1p
    data_log = np.log1p(data_cpm.values)

    # per‑gene z‑score
    mean = data_log.mean(axis=1, keepdims=True)
    std = data_log.std(axis=1, keepdims=True)
    data_z = (data_log - mean) / std

    # Transpose → samples×features  (cells×genes)
    X = data_z.T.astype("float32")
    y = anno.astype("str")

    if limit is not None and limit < X.shape[0]:
        X, y = X[: limit], y[: limit]
    return X, y

def get_args():
    p = argparse.ArgumentParser("DAE → GraphDR‑NN pipeline")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--limit", type=int, default=10000)
    # DAE
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--dae-epochs", type=int, default=1000)
    p.add_argument("--dae-noise", type=float, default=0.2)
    # misc
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()
def train_dae(X: np.ndarray, latent_dim: int, epochs: int, batch: int, noise: float, lr: float, device: str):
    N, D = X.shape
    dae = DAE(D, latent_dim).to(device)
    opt = torch.optim.Adam(dae.parameters(), lr=lr)
    mse = nn.MSELoss()

    loader = DataLoader(torch.as_tensor(X), batch_size=batch, shuffle=True)
    for ep in range(1, epochs + 1):
        s = 0.0
        for xb in loader:
            xb = xb.to(device)
            noisy = xb + noise * torch.randn_like(xb)
            recon, _ = dae(noisy)
            loss = mse(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            s += loss.item() * xb.size(0)
        if ep % 10 == 0 or ep == 1:
            print(f"DAE {ep:3d}/{epochs}  MSE: {s / N:.4f}")

    return  torch.as_tensor(X, device=device)



def graphdr(X, n_neighbors=15, lambda_reg=1.0, N_Components=2):
    n_samples = X.shape[0]

    # 1. build KNN graph
    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    
    # 2. graph Laplace
    L = csgraph.laplacian(knn_graph, normed=False)

    # 3. smooth
    I = eye(n_samples)
    A = I + lambda_reg * L

    cho = cho_factor(A.toarray())
    X_smooth = cho_solve(cho, X)
    pca = PCA(n_components=N_Components)
    Z = pca.fit_transform(X_smooth)
    return Z




def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = load_dataset(limit=args.limit)
    print(f"Dataset loaded: N={X.shape[0]}, D={X.shape[1]}")

    latent = train_dae(
        X,
        latent_dim=args.latent_dim,
        epochs=args.dae_epochs,
        batch=args.batch,
        lr=args.lr,
        device=device,
        noise=args.dae_noise
    )
    # np.save("latent.npy", latent)
    latent = latent.cpu().numpy()
    X = graphdr(latent, n_neighbors=15, lambda_reg=0.01, N_Components=2)
    if y is not None:
        np.save("labels.npy", y)
    print("Saved latent.npy and embeddings.npy")
    plt.figure(figsize=(15,10))
    seaborn.scatterplot(x=X[:,0], y=X[:,1], linewidth = 0, s=3, hue=y)
    plt.xlabel('GraphDR 1')
    plt.ylabel('GraphDR 2')
    plt.savefig('deepGraphDR_scatter.png')


if __name__ == "__main__":
    main()
