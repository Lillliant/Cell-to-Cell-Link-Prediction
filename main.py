"""Main training script for cell-cell link prediction experiments."""

import argparse
import json
import os
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
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
    plot_training_history,
    set_seed,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(data_path, data_name, k_neighbors):
    """Load one dataset into a PyG Data object."""
    print(f"[LOAD] Preparing dataset '{data_name}' from '{data_path}'")
    if data_name == "tonsil":
        print("[LOAD] Using tonsil h5ad loader")
        data = load_h5ad_graph(os.path.join(data_path, "human_tonsil_slidetags.h5ad"), k_neighbors)
        print(f"[LOAD] Loaded tonsil graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
        return data
    if data_name == "myocardial_infarction":
        print("[LOAD] Using myocardial infarction h5ad loader")
        data = load_h5ad_graph(os.path.join(data_path, "Visium_control_P1.h5ad"), k_neighbors)
        print(f"[LOAD] Loaded myocardial_infarction graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
        return data

    folder_path = os.path.join(data_path, data_name)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"[LOAD] Using folder loader for '{folder_path}'")
        data = load_pancreas_folder(folder_path)
        print(f"[LOAD] Loaded folder graph with {data.num_nodes} nodes and {data.edge_index.size(1)} edges")
        return data

    raise ValueError(
        "Unsupported data_name. Use one of: tonsil, myocardial_infarction, "
        "or a folder under data_path such as human_pancreas_HumanD3."
    )


def preprocess(data):
    """Normalize tensor dtypes for model training."""
    print("[PREPROCESS] Casting edge_index to long and features to float")
    data.edge_index = data.edge_index.long()
    data.x = data.x.float()
    print(f"[PREPROCESS] Done. x shape: {tuple(data.x.shape)}, edge_index shape: {tuple(data.edge_index.shape)}")
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


def evaluate_split(model, split_data, criterion, hyperedge_index=None):
    """Evaluate one data split and return losses/scores/probabilities."""
    model.eval()
    with torch.no_grad():
        z = model(split_data.x, split_data.edge_index, hyperedge_index)
        logits = decode(z, split_data.edge_label_index)
        loss = criterion(logits, split_data.edge_label.float()).item()
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = split_data.edge_label.cpu().numpy()

    # AUC is undefined if all labels are the same class in this split.
    auc = roc_auc_score(y_true, probs) if len(set(y_true.tolist())) > 1 else float("nan")
    ap = average_precision_score(y_true, probs)
    y_pred = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "loss": float(loss),
        "auc": float(auc),
        "ap": float(ap),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "probs": probs,
        "y_true": y_true,
    }


def evaluate(model, split_data, criterion, hyperedge_index=None):
    """Evaluate model with ROC/PR curves and classification metrics."""
    print("[EVAL] Running evaluation on test edges")
    metrics = evaluate_split(model, split_data, criterion, hyperedge_index)
    probs = metrics["probs"]
    y_true = metrics["y_true"]
    if len(set(y_true.tolist())) > 1:
        fpr, tpr, _ = roc_curve(y_true, probs)
    else:
        # Guard against degenerate splits with only one label class.
        fpr, tpr = [0.0, 1.0], [0.0, 1.0]
    precision, recall, _ = precision_recall_curve(y_true, probs)
    print(
        "[EVAL] Complete. "
        f"AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}, "
        f"Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
    )
    return metrics, fpr, tpr, precision, recall


def train(model, train_data, val_data, test_data, cfg, hyperedge_index=None):
    """Train one model and return its test metrics."""
    print(f"[TRAIN] Starting training for {cfg.epochs} epochs on device: {device}")
    print(
        "[TRAIN] Train edges: "
        f"{train_data.edge_index.size(1)}, labeled train pairs: {train_data.edge_label_index.size(1)}, "
        f"labeled test pairs: {test_data.edge_label_index.size(1)}"
    )
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
    # BCEWithLogitsLoss expects raw scores (logits), so we do NOT apply sigmoid before loss.
    criterion = nn.BCEWithLogitsLoss()
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 1) Encode each node into an embedding vector.
        z = model(train_data.x, train_data.edge_index, hyperedge_index)
        # 2) Score each candidate edge by comparing its endpoint embeddings.
        logits = decode(z, train_data.edge_label_index)
        loss = criterion(logits, train_data.edge_label.float())
        # Standard optimization step: backprop gradients then update model weights.
        loss.backward()
        optimizer.step()

        # Monitor both train and validation behavior every epoch.
        train_metrics = evaluate_split(model, train_data, criterion, hyperedge_index)
        val_metrics = evaluate_split(model, val_data, criterion, hyperedge_index)
        history["epochs"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_auc"].append(val_metrics["auc"])

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_metrics['loss']:.4f} | Val loss: {val_metrics['loss']:.4f} | "
                f"Train AUC: {train_metrics['auc']:.4f} | Val AUC: {val_metrics['auc']:.4f}"
            )

    print("[TRAIN] Training loop complete, starting evaluation")
    test_metrics, fpr, tpr, precision, recall = evaluate(model, test_data, criterion, hyperedge_index)
    print("[TRAIN] Model run complete")
    return {
        "loss": test_metrics["loss"],
        "auc": test_metrics["auc"],
        "ap": test_metrics["ap"],
        "accuracy": test_metrics["accuracy"],
        "precision_at_05": test_metrics["precision"],
        "recall_at_05": test_metrics["recall"],
        "f1_at_05": test_metrics["f1"],
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "history": history,
    }


def run_models(data, data_name, model_names, output_path, cfg):
    """Run a model list on the same train/test split and save per-model metrics."""
    print(f"[SPLIT] Creating train/test split for dataset '{data_name}'")
    transform = RandomLinkSplit(
        num_val=cfg.val_ratio,
        num_test=cfg.test_ratio,
        is_undirected=True,
        # Adds negative (non-edge) examples so link prediction is a binary task.
        add_negative_train_samples=True,
        neg_sampling_ratio=cfg.neg_sampling_ratio,
    )
    train_data, val_data, test_data = transform(data)
    print(
        "[SPLIT] Done. "
        f"Train message edges: {train_data.edge_index.size(1)}, "
        f"train labels: {train_data.edge_label_index.size(1)}, "
        f"val labels: {val_data.edge_label_index.size(1)}, "
        f"test labels: {test_data.edge_label_index.size(1)}"
    )

    all_results = []
    out_dir = os.path.join(output_path, data_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[OUTPUT] Saving per-model artifacts to '{out_dir}'")

    for model_name in model_names:
        print(f"\n===== Training {model_name} on {data_name} =====")
        print(f"[MODEL] Initializing model '{model_name}'")
        model = get_model(model_name, in_channels=data.x.size(1), cfg=cfg)

        hyperedge_index = None
        if model_name.lower() == "hgnn":
            print("[MODEL] Building hyperedge index for HGNN")
            # HGNN needs node->hyperedge membership instead of pairwise edges.
            hyperedge_index = build_hyperedge_index(train_data.edge_index, num_nodes=train_data.num_nodes)

        result = train(model, train_data, val_data, test_data, cfg, hyperedge_index=hyperedge_index)
        print(f"[PLOT] Writing ROC/PR curves for '{model_name}'")
        plot_curves(result, model_name, data_name, out_dir)
        print(f"[PLOT] Writing train/validation history plots for '{model_name}'")
        plot_training_history(result["history"], model_name, data_name, out_dir)

        summary = {
            "dataset": data_name,
            "model": model_name,
            "auc": result["auc"],
            "ap": result["ap"],
            "accuracy": result["accuracy"],
            "precision_at_05": result["precision_at_05"],
            "recall_at_05": result["recall_at_05"],
            "f1_at_05": result["f1_at_05"],
            "test_loss": result["loss"],
        }
        all_results.append(summary)
        print(
            f"{model_name} | AUC: {result['auc']:.4f} | AP: {result['ap']:.4f} | "
            f"Accuracy: {result['accuracy']:.4f} | Precision: {result['precision_at_05']:.4f} | "
            f"Recall: {result['recall_at_05']:.4f} | F1: {result['f1_at_05']:.4f}"
        )

    print(f"[OUTPUT] Writing metrics tables to '{out_dir}'")
    pd.DataFrame(all_results).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OUTPUT] Finished dataset '{data_name}'")

    return all_results


def main(data_path, data_names, model_names, output_path, config):
    """Run end-to-end experiments for all selected datasets and models."""
    print("[PIPELINE] Starting experiment pipeline")
    set_seed(config.seed)
    print(f"[PIPELINE] Seed set to {config.seed}")
    print(f"[PIPELINE] Datasets: {data_names}")
    print(f"[PIPELINE] Models: {model_names}")
    print(f"[PIPELINE] Output path: {output_path}")
    all_results = []

    for data_name in data_names:
        print(f"\n[DATASET] Starting dataset '{data_name}'")
        data = load(data_path, data_name, k_neighbors=config.k_neighbors)
        data = preprocess(data)
        results = run_models(data, data_name, model_names, output_path, config)
        all_results.extend(results)
        print(f"[DATASET] Completed dataset '{data_name}'")

    os.makedirs(output_path, exist_ok=True)
    summary_path = os.path.join(output_path, "summary_metrics.csv")
    print(f"[OUTPUT] Writing summary metrics to '{summary_path}'")
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print("[PIPELINE] Finished. Saved summary metrics to", summary_path)


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
    parser.add_argument("--val_ratio", type=float, help="Validation split ratio for fit diagnostics")
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
        val_ratio=args.val_ratio if args.val_ratio is not None else base_cfg.val_ratio,
    )

    data_names = [name.strip() for name in data_names_arg.split(",") if name.strip()]
    model_names = [name.strip() for name in models_arg.split(",") if name.strip()]
    main(data_path, data_names, model_names, output_path, cfg)
