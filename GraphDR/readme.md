# GraphDR Demonstration

This repository contains a demonstration of the **GraphDR** algorithm, a linearized graph-based dimensionality reduction technique designed for structure-preserving visualization of high-dimensional data such as single-cell RNA sequencing (scRNA-seq).

GraphDR blends the strengths of PCA and graph smoothing into a unified framework for denoising and dimensionality reduction.

---

## 📘 What’s in This Notebook?

The `GraphDR_test.ipynb` notebook includes:

- Construction of a k-NN graph from input data
- Graph smoothing via linear system solving
- Dimensionality reduction via PCA
- Visualization of smoothed data
- Example with MNIST or toy data (depending on your dataset)

---

## 🔧 Requirements

You can install the dependencies using:

```bash
pip install numpy scipy scikit-learn matplotlib