"""
Evaluation: classification report, confusion matrix, PR curve, failure analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from .cam import generate_cam_comparison, save_failure_cases
from .dataset import CLASS_NAMES, IDX_TO_CLASS, get_dataloaders
from .model import BackboneName, build_model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Run inference on a split and return predictions."""
    model.eval()
    all_labels, all_preds, all_probs, all_paths = [], [], [], []

    for images, labels, paths in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.vstack(all_probs)

    pneumonia_probs = probs[:, 1]
    pr_auc = average_precision_score(labels, pneumonia_probs)
    try:
        roc_auc = roc_auc_score(labels, pneumonia_probs)
    except ValueError:
        roc_auc = float("nan")

    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True
    )

    return {
        "labels": labels,
        "preds": preds,
        "probs": probs,
        "paths": all_paths,
        "classification_report": report,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
    pr_auc = average_precision_score(labels, probs[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Pneumonia)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    auc = roc_auc_score(labels, probs[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Pneumonia)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history_path: Path,
    save_path: Path,
) -> None:
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key, ax, title in [
        ("f1", axes[0], "F1 Score"),
        ("loss", axes[1], "Loss"),
    ]:
        ax.plot(epochs, [m[key] for m in history["train"]], label="Train")
        ax.plot(epochs, [m[key] for m in history["val"]], label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.upper())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_evaluation(
    checkpoint_path: str | Path,
    data_root: str | Path = "data/raw/chest_xray",
    output_dir: str | Path = "reports/figures",
    split: str = "test",
    batch_size: int = 32,
    generate_cams: bool = True,
    num_cam_samples: int = 12,
    num_failure_cases: int = 6,
) -> dict:
    """Full evaluation pipeline with visualizations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone: BackboneName = checkpoint.get("backbone", "resnet50")

    model = build_model(backbone=backbone, num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    loaders = get_dataloaders(data_root, batch_size=batch_size, num_workers=0)
    results = evaluate_model(model, loaders[split], device)

    # Save metrics JSON
    metrics = {
        "classification_report": results["classification_report"],
        "pr_auc": results["pr_auc"],
        "roc_auc": results["roc_auc"],
        "confusion_matrix": results["confusion_matrix"],
    }
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(classification_report(
        results["labels"],
        results["preds"],
        target_names=CLASS_NAMES,
    ))
    print(f"PR AUC: {results['pr_auc']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")

    cm = np.array(results["confusion_matrix"])
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    plot_precision_recall(
        results["labels"], results["probs"], output_dir / "precision_recall_curve.png"
    )
    plot_roc_curve(results["labels"], results["probs"], output_dir / "roc_curve.png")

    history_path = checkpoint_path.parent / f"history_{backbone}.json"
    if history_path.exists():
        plot_training_curves(history_path, output_dir / "training_curves.png")

    if generate_cams:
        cam_dir = output_dir / "cam_comparisons"
        cam_dir.mkdir(exist_ok=True)
        generate_cam_comparison(
            model=model,
            loader=loaders[split],
            device=device,
            backbone=backbone,
            output_dir=cam_dir,
            num_samples=num_cam_samples,
        )

        failure_dir = output_dir / "failure_cases"
        save_failure_cases(
            model=model,
            labels=results["labels"],
            preds=results["preds"],
            paths=results["paths"],
            loader=loaders[split],
            device=device,
            backbone=backbone,
            output_dir=failure_dir,
            num_cases=num_failure_cases,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pneumonia classifier")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/raw/chest_xray")
    parser.add_argument("--output-dir", type=str, default="reports/figures")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--no-cams", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split=args.split,
        generate_cams=not args.no_cams,
    )


if __name__ == "__main__":
    main()
