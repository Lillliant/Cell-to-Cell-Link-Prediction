"""Main training script for cell-cell link prediction experiments."""

import argparse
import json
import os
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch import nn
from torch_geometric.transforms import RandomLinkSplit

from models.GAT import GATEncoder
from models.GCN import GCNEncoder
from models.GNN import GNNEncoder
from models.GraphTransformer import GraphTransformerEncoder
from models.HGNN import HGNNEncoder
from param import TrainingConfig
from util.util import (
    build_hyperedge_index,
    decode,
    load_h5ad_graph,
    load_pancreas_folder,
    plot_curves,
    set_seed,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(data_path, data_name, k_neighbors):
    """Load one dataset into a PyG Data object."""
    if data_name == "tonsil":
        return load_h5ad_graph(os.path.join(data_path, "human_tonsil_slidetags.h5ad"), k_neighbors)
    if data_name == "myocardial_infarction":
        return load_h5ad_graph(os.path.join(data_path, "Visium_control_P1.h5ad"), k_neighbors)

    folder_path = os.path.join(data_path, data_name)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return load_pancreas_folder(folder_path)

    raise ValueError(
        "Unsupported data_name. Use one of: tonsil, myocardial_infarction, "
        "or a folder under data_path such as human_pancreas_HumanD3."
    )


def preprocess(data):
    """Normalize tensor dtypes for model training."""
    data.edge_index = data.edge_index.long()
    data.x = data.x.float()
    return data


def get_model(model_name, in_channels, cfg):
    """Instantiate an encoder by model name."""
    model_name = model_name.lower()
    if model_name == "gnn":
        return GNNEncoder(in_channels, cfg.hidden_channels, cfg.out_channels, cfg.dropout)
    if model_name == "gcn":
        return GCNEncoder(in_channels, cfg.hidden_channels, cfg.out_channels, cfg.dropout)
    if model_name == "gat":
        return GATEncoder(in_channels, cfg.hidden_channels, cfg.out_channels, cfg.dropout, cfg.heads)
    if model_name == "hgnn":
        return HGNNEncoder(in_channels, cfg.hidden_channels, cfg.out_channels, cfg.dropout)
    if model_name in {"graph_transformer", "graphtransformer", "gt"}:
        return GraphTransformerEncoder(in_channels, cfg.hidden_channels, cfg.out_channels, cfg.dropout, cfg.heads)
    raise ValueError(f"Unknown model name: {model_name}")


def evaluate(model, split_data, hyperedge_index=None):
    """Evaluate model with ROC-AUC and Average Precision."""
    model.eval()
    with torch.no_grad():
        z = model(split_data.x, split_data.edge_index, hyperedge_index)
        logits = decode(z, split_data.edge_label_index)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = split_data.edge_label.cpu().numpy()

    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    return auc, ap, fpr, tpr, precision, recall


def train(model, train_data, test_data, cfg, hyperedge_index=None):
    """Train one model and return its test metrics."""
    model = model.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    if hyperedge_index is not None:
        hyperedge_index = hyperedge_index.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model(train_data.x, train_data.edge_index, hyperedge_index)
        logits = decode(z, train_data.edge_label_index)
        loss = criterion(logits, train_data.edge_label.float())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    auc, ap, fpr, tpr, precision, recall = evaluate(model, test_data, hyperedge_index)
    return {
        "auc": float(auc),
        "ap": float(ap),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


def run_models(data, data_name, model_names, output_path, cfg):
    """Run a model list on the same train/test split and save per-model metrics."""
    transform = RandomLinkSplit(
        num_val=0.0,
        num_test=cfg.test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=cfg.neg_sampling_ratio,
    )
    train_data, _, test_data = transform(data)

    all_results = []
    out_dir = os.path.join(output_path, data_name)
    os.makedirs(out_dir, exist_ok=True)

    for model_name in model_names:
        print(f"\n===== Training {model_name} on {data_name} =====")
        model = get_model(model_name, in_channels=data.x.size(1), cfg=cfg)

        hyperedge_index = None
        if model_name.lower() == "hgnn":
            hyperedge_index = build_hyperedge_index(train_data.edge_index, num_nodes=train_data.num_nodes)

        result = train(model, train_data, test_data, cfg, hyperedge_index=hyperedge_index)
        plot_curves(result, model_name, data_name, out_dir)

        summary = {
            "dataset": data_name,
            "model": model_name,
            "auc": result["auc"],
            "ap": result["ap"],
        }
        all_results.append(summary)
        print(f"{model_name} | AUC: {result['auc']:.4f} | AP: {result['ap']:.4f}")

    pd.DataFrame(all_results).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main(data_path, data_names, model_names, output_path, config):
    """Run end-to-end experiments for all selected datasets and models."""
    set_seed(config.seed)
    all_results = []

    for data_name in data_names:
        data = load(data_path, data_name, k_neighbors=config.k_neighbors)
        data = preprocess(data)
        results = run_models(data, data_name, model_names, output_path, config)
        all_results.extend(results)

    os.makedirs(output_path, exist_ok=True)
    summary_path = os.path.join(output_path, "summary_metrics.csv")
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print("\nSaved summary metrics to", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path containing datasets")
    parser.add_argument(
        "--data_names",
        type=str,
        help="Comma-separated dataset names. For pancreas use folder names under data_path.",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names: gnn,gcn,gat,hgnn,graph_transformer",
    )
    parser.add_argument("--output_path", type=str, help="Path to save results and evaluations")

    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--hidden_channels", type=int, help="Hidden layer size")
    parser.add_argument("--out_channels", type=int, help="Output embedding size")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--heads", type=int, help="Attention heads for GAT/GraphTransformer")
    parser.add_argument("--k_neighbors", type=int, help="kNN neighbors for h5ad graph construction")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    base_cfg = TrainingConfig()
    data_path = args.data_path if args.data_path is not None else "data"
    data_names_arg = args.data_names if args.data_names is not None else "human_pancreas_HumanD3,tonsil,myocardial_infarction"
    models_arg = args.models if args.models is not None else "gnn,gcn,gat,hgnn,graph_transformer"
    output_path = args.output_path if args.output_path is not None else "outputs"

    cfg = TrainingConfig(
        seed=args.seed if args.seed is not None else base_cfg.seed,
        hidden_channels=args.hidden_channels if args.hidden_channels is not None else base_cfg.hidden_channels,
        out_channels=args.out_channels if args.out_channels is not None else base_cfg.out_channels,
        dropout=args.dropout if args.dropout is not None else base_cfg.dropout,
        heads=args.heads if args.heads is not None else base_cfg.heads,
        epochs=args.epochs if args.epochs is not None else base_cfg.epochs,
        k_neighbors=args.k_neighbors if args.k_neighbors is not None else base_cfg.k_neighbors,
    )

    data_names = [name.strip() for name in data_names_arg.split(",") if name.strip()]
    model_names = [name.strip() for name in models_arg.split(",") if name.strip()]
    main(data_path, data_names, model_names, output_path, cfg)
