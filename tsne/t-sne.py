import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def save_metadata_tsv(labels, filepath="runs/tsne_mnist/metadata.tsv"):
    """
    generate a tsv file to match the color and label
    """
    with open(filepath, "w") as f:
        for label in labels:
            f.write(f"{label}\n")

def plot_tsne_scatter(Y, labels):
    """
    Plot a 2D scatter plot for t-SNE embeddings.

    Args:
        Y: ndarray or Tensor of shape (N, 2) - 2D embeddings
        labels: ndarray or Tensor of shape (N,) - class labels
    """
    if torch.is_tensor(Y): Y = Y.cpu().detach().numpy()
    if torch.is_tensor(labels): labels = labels.cpu().detach().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)


    handles = []
    for i in np.unique(labels):
        handles.append(plt.Line2D([], [], marker='o', linestyle='', 
                                  label=str(i), color=plt.cm.tab10(i/10), markersize=6))
    ax.legend(handles=handles, title="Digit Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("t-SNE on MNIST (with label legend)")
    ax.axis('off')
    plt.tight_layout()
    return fig

writer = SummaryWriter(log_dir="runs/tsne_mnist")  # tensorboard file address

def Hbeta(D, beta=1.0):
    """
    Compute entropy(H) and probability(P) from nxn distance matrix.

    Parameters
    ----------
    D : numpy.ndarray
        distance matrix (n,n)
    beta : float
        precision measure
    .. math:: \beta = \frac{1}/{(2 * \sigma^2)}

    Returns
    -------
    H : float
        entropy
    P : numpy.ndarray
        probability matrix (n,n)
    """
    num = np.exp(-D * beta)
    den = np.sum(np.exp(-D * beta), 0)
    P = num / den
    H = np.log(den) + beta * np.sum(D * num) / (den)
    return H, P


def adjustbeta(X, D,tol, perplexity):
    """
    dichotomy search for the best beta
    Precision(beta) adjustment based on perplexity

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    tol : float
        tolerance for the stopping criteria of beta adjustment
    perplexity : float
        perplexity can be interpreted as a smooth measure of the effective number of neighbors

    Returns
    -------
    P : numpy.ndarray
        probability matrix (n,n)
    beta : numpy.ndarray
        precision array (n,1)
    """
    (n, d) = X.shape
    # Need to compute D here, which is nxn distance matrix of X
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisP

    return P, beta


def pairwise_distances(x):
    """
    Compute squared Euclidean distances between rows of x.

    Args:
        x: Tensor of shape (n, d)

    Returns:
        Tensor of shape (n, n) - distance matrix
    """
    sum_x = torch.sum(x ** 2, dim=1, keepdim=True)
    return sum_x + sum_x.T - 2 * x @ x.T

def high_dim_similarity(X, perplexity=30.0, tol=1e-5):
    """
    Compute high-dimensional pairwise similarities.

    Args:
        X: Input tensor (n, d)
        perplexity: Effective neighborhood size
        tol: Search tolerance

    Returns:
        P: Symmetric similarity matrix (n, n)
    """
    D = pairwise_distances(X).cpu().numpy()
    X_np = X.detach().cpu().numpy()
    P, beta = adjustbeta(X_np, D, tol, perplexity)
    P = (P + P.T) / (2.0 * X.shape[0])  # 对称归一化
    return torch.tensor(P, dtype=torch.float32, device=X.device)

def compute_low_dim_similarities(Y):
    """
    Compute low-dimensional similarities using t-distribution kernel.

    Args:
        Y: 2D embeddings (n, d)

    Returns:
        Q: Similarity matrix (n, n)
    """
    N = Y.shape[0]
    D = pairwise_distances(Y)
    inv = (1.0 + D)**-1 #t distribution
    inv.fill_diagonal_(0)
    Q = inv / torch.sum(inv)
    return Q

def kl_divergence(P, Q):
    """
    Compute Kullback–Leibler divergence between P and Q.

    Args:
        P, Q: Probability matrices (n, n)

    Returns:
        Scalar loss value
    """
    return torch.sum(P[P>0] * torch.log(P[P>0] / (Q[P>0]+1e-10)))

def tsne(X,labels, n_components=2, perplexity=30.0, tol=1e-5, learning_rate=2.0, n_iter=1000):
    """
    Run t-SNE on input tensor X.

    Args:
        X: Input data (n, d)
        labels: Corresponding labels (n,)
        n_components: Target dimension (default: 2)
        perplexity: Controls local/global trade-off
        tol: Tolerance for beta search
        learning_rate: Optimizer learning rate
        n_iter: Number of optimization steps

    Returns:
        Numpy array of shape (n, n_components)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    N, D = X.shape
    #calculate high-dim similarity
    # X = (X-X.mean(dim=0))/(X.std(dim=0)+1e-10) #normalize
    P = high_dim_similarity(X, perplexity, tol)
    X = X.to(device)
    P = P.to(device)
    # initialize low-dim embeddings
    Y = torch.randn(N, n_components, device=device)
    Y.requires_grad = True
    optimizer = optim.Adam([Y], lr=learning_rate)
    for i in range(n_iter):
        optimizer.zero_grad()
        Q = compute_low_dim_similarities(Y)
        loss = kl_divergence(P, Q)  
        loss.backward()
        optimizer.step()
        if i % 10 == 0 or i == n_iter - 1:
            print(f"Iteration {i}, Loss: {loss.item()}")
            writer.add_scalar("KL Divergence", loss.item(), i)
    if labels is not None:
        writer.add_embedding(Y.detach().cpu(), metadata=labels, tag="MNIST-tsne")
        fig = plot_tsne_scatter(Y.detach().cpu(), labels)
        writer.add_figure("t-SNE Scatter", fig, global_step=i)
    writer.close()
    return Y.detach().cpu().numpy()

if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist, batch_size=1000, shuffle=True)
    images, labels = next(iter(data_loader))
    X_raw = images.view(images.shape[0], -1)
    X_tensor = X_raw
    # Run t-SNE
    save_metadata_tsv(labels)
    tsne(X_tensor, labels)
    print("t-SNE done")

        
    



