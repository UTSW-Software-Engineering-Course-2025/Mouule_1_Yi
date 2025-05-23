# Yi's Module 1

This repository demonstrates and compares two powerful dimensionality reduction methods implemented from scratch using Python:

- 🔹 **GraphDR** – a linear graph-regularized smoother for structured embedding
- 🔹 **t-SNE** – a non-linear, probabilistic method for manifold visualization

Both methods are applied to high-dimensional data like MNIST or single-cell datasets to explore geometry and clustering in low dimensions.

---

## Submodules Overview

| Module    | Description                                                  | Link                                     |
| --------- | ------------------------------------------------------------ | ---------------------------------------- |
| 🧠 GraphDR | Graph-based dimensionality reduction using Laplacian smoothing + PCA | [GraphDR/README.md](./GraphDR/readme.md) |
| 🔍 t-SNE   | From-scratch PyTorch implementation of t-SNE with TensorBoard support | [tsne/README.md](./tsne/README.md)       |