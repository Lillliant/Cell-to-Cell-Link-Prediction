import os
import numpy as np
import scanpy as sc
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.transforms import RandomLinkSplit

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(SEED)

def load(data_path: str, data_name: str):
    if data_name == 'tonsil':
        adata = sc.read_h5ad(os.path.join(data_path, "human_tonsil_slidetags.h5ad"))
    elif data_name == 'myocardial_infarction':
        adata = sc.read_h5ad(os.path.join(data_path, "Visium_control_P1.h5ad"))

def preprocess():
    pass

def train():
    pass

def evaluate():
    pass

def main(data_path: str, data_name: str, output_path: str):
    data = load(data_path, data_name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/", help="Path to the dataset as a .h5ad file")
    parser.add_argument("--data_name", type=str, default="dataset_name", help="Name of the dataset for logging and saving results")
    parser.add_argument("--output_path", type=str, default="outputs", help="Path to save results and evaluations")

    args = parser.parse_args()

    main(args.data_path)