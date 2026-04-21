"""Utility functions for data loading, preprocessing, and plotting."""

import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_dense_array(x):
    """Convert sparse/dense matrix-like objects to a NumPy array."""
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def read_first_existing(paths):
    """Return the first existing path in a list, otherwise raise."""
    for path in paths:
        if path is not None and os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any expected files: {paths}")


def load_edge_list(folder_path):
    """Load an edge list from csv/txt/edgelist files."""
    edge_candidates = [
        next(iter(glob.glob(os.path.join(folder_path, "edgelist_encoded_*.csv"))), None),
        next(iter(glob.glob(os.path.join(folder_path, "edgelist_encoded_*.txt"))), None),
        next(iter(glob.glob(os.path.join(folder_path, "edgelist_encoded_*.edgelist"))), None),
        next(iter(glob.glob(os.path.join(folder_path, "Weighted_edgelist_encoded_*.edgelist"))), None),
    ]
    edge_path = read_first_existing(edge_candidates)

    if os.path.splitext(edge_path)[1] == ".csv":
        df = pd.read_csv(edge_path)
        if {"source", "target"}.issubset(df.columns):
            src = df["source"].to_numpy()
            dst = df["target"].to_numpy()
        else:
            src = df.iloc[:, 0].to_numpy()
            dst = df.iloc[:, 1].to_numpy()
    else:
        df = pd.read_csv(edge_path, sep=r"\s+", header=None)
        src = df.iloc[:, 0].to_numpy()
        dst = df.iloc[:, 1].to_numpy()

    src = pd.to_numeric(src, errors="coerce").astype(np.int64)
    dst = pd.to_numeric(dst, errors="coerce").astype(np.int64)

    # Some provided edge lists are 1-indexed.
    if src.min() >= 1 and dst.min() >= 1:
        src = src - 1
        dst = dst - 1

    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


def load_pancreas_folder(folder_path):
    """Load a pancreas graph folder containing node features and edge list."""
    attr_candidates = [
        next(iter(glob.glob(os.path.join(folder_path, "attributes_IG_*.csv"))), None),
        next(iter(glob.glob(os.path.join(folder_path, "*feature_selected*.csv"))), None),
    ]
    attr_path = read_first_existing(attr_candidates)

    attr_df = pd.read_csv(attr_path)
    if attr_df.shape[1] > 1:
        features = attr_df.select_dtypes(include=[np.number]).to_numpy()
    else:
        features = attr_df.to_numpy()
    features = np.nan_to_num(features, nan=0.0)

    edge_index = load_edge_list(folder_path)
    num_nodes = max(features.shape[0], int(edge_index.max().item()) + 1)

    if features.shape[0] < num_nodes:
        pad_rows = num_nodes - features.shape[0]
        features = np.vstack([features, np.zeros((pad_rows, features.shape[1]), dtype=features.dtype)])

    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index)


def load_h5ad_graph(h5ad_path, k_neighbors):
    """Load and preprocess h5ad data, then build a kNN graph."""
    adata = sc.read_h5ad(str(h5ad_path))

    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_top_genes = min(2000, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    if "highly_variable" in adata.var:
        adata = adata[:, adata.var["highly_variable"]].copy()

    x = to_dense_array(adata.X).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)

    adjacency = kneighbors_graph(
        x,
        n_neighbors=min(k_neighbors, max(2, x.shape[0] - 1)),
        mode="connectivity",
        include_self=False,
    )
    edge_index, _ = from_scipy_sparse_matrix(adjacency)
    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)


def build_hyperedge_index(edge_index, num_nodes):
    """Build a simple hyperedge incidence matrix from node neighborhoods."""
    src, dst = edge_index
    neighborhoods = [[] for _ in range(num_nodes)]
    for s, d in zip(src.tolist(), dst.tolist()):
        neighborhoods[s].append(d)
        neighborhoods[d].append(s)

    node_ids = []
    hyperedge_ids = []
    for center in range(num_nodes):
        members = set(neighborhoods[center])
        members.add(center)
        for node in members:
            node_ids.append(node)
            hyperedge_ids.append(center)

    return torch.tensor([node_ids, hyperedge_ids], dtype=torch.long)


def decode(z, edge_label_index):
    """Dot-product decoder for edge prediction."""
    src, dst = edge_label_index
    return (z[src] * z[dst]).sum(dim=-1)


def plot_curves(result, model_name, dataset_name, output_dir):
    """Save ROC and Precision-Recall plots for one model run."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(result["fpr"], result["tpr"], linestyle="--", label=f"{model_name} (AUC={result['auc']:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve for {dataset_name} - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_{dataset_name}_{model_name}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(result["recall"], result["precision"], linestyle="--", label=f"{model_name} (AP={result['ap']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall for {dataset_name} - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pr_{dataset_name}_{model_name}.png"), dpi=150)
    plt.close()


def plot_training_history(history, model_name, dataset_name, output_dir):
    """Save train/validation loss and metric curves for one model run.

    The paired train/validation curves are useful for diagnosing underfitting
    and overfitting behavior across epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = history["epochs"]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title(f"Loss history for {dataset_name} - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_history_{dataset_name}_{model_name}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_auc"], label="Train AUC")
    plt.plot(epochs, history["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"AUC history for {dataset_name} - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"auc_history_{dataset_name}_{model_name}.png"), dpi=150)
    plt.close()
