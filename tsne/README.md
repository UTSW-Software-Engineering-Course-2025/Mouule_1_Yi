# Torch t-SNE on MNIST

This project provides a clean, from-scratch implementation of t-SNE (t-Distributed Stochastic Neighbor Embedding) using **PyTorch**. It demonstrates how to reduce high-dimensional data (like MNIST images) into 2D for visualization purposes.


## Features

- Manual computation of pairwise distances and probability matrices (P and Q)
- Custom implementation of KL divergence as loss
- Optimization via PyTorch's autograd and Adam optimizer
- TensorBoard integration for:
  - Loss curves
  - 2D embedding visualization (projector tab)
- Clean visualization of digit clusters in 2D

---

## File Structure

├── t-sne.py             # Main script
 ├── tsne_mnist/                # TensorBoard logs and embeddings
 ├── raw/                # Downloaded MNIST dataset
 └── README.md

### Run the script

```bash
python t-sne.py
```

This will:

- Download the MNIST dataset
- Flatten the 28x28 images into 784-dimensional vectors
- Run t-SNE to embed them into 2D space
- Save TensorBoard visualizations

### Launch TensorBoard

```bash
tensorboard --logdir=runs/tsne_mnist
```

Go to http://localhost:6006 to see:

- The KL divergence curve
- Embedding visualization under the **"Projector"** tab