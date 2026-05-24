"""
Training loop with weighted loss, F1-based checkpointing, and LR scheduling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .dataset import (
    CLASS_NAMES,
    compute_class_weights,
    get_dataloaders,
    get_dataset_stats,
)
from .model import BackboneName, build_model, count_parameters, unfreeze_layers


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = _compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels, _ in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.vstack(all_probs)

    metrics = _compute_metrics(labels_arr, preds_arr)
    metrics["loss"] = epoch_loss
    return metrics, labels_arr, preds_arr, probs_arr


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = (y_true == y_pred).mean()
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
    }


def train(
    data_root: str | Path,
    backbone: BackboneName = "resnet50",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    image_size: int = 224,
    dropout: float = 0.5,
    output_dir: str | Path = "models",
    use_class_weights: bool = True,
    use_weighted_sampler: bool = True,
    unfreeze_epoch: int = 5,
    num_workers: int = 4,
) -> dict:
    """Full training pipeline. Returns history dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset stats: {json.dumps(get_dataset_stats(data_root), indent=2)}")

    loaders = get_dataloaders(
        data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler,
    )

    train_labels = loaders["train"].dataset.labels
    class_weights = compute_class_weights(train_labels)
    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"Class weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    model = build_model(
        backbone=backbone,
        num_classes=2,
        pretrained=True,
        dropout=dropout,
        freeze_backbone=True,
    )
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    model = model.to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train": [], "val": []}
    best_f1 = 0.0
    best_path = output_dir / f"best_{backbone}.pth"

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch:
            print(f"Epoch {epoch}: Unfreezing later layers for fine-tuning...")
            unfreeze_layers(model, backbone, layer_idx=4 if backbone == "resnet50" else 5)
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 0.1
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_metrics, _, _, _ = validate(
            model, loaders["val"], criterion, device
        )
        scheduler.step(val_metrics["f1"])

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_f1,
                    "backbone": backbone,
                    "class_names": CLASS_NAMES,
                },
                best_path,
            )
            print(f"  -> Saved best model (F1={best_f1:.4f})")

    history_path = output_dir / f"history_{backbone}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")
    print(f"Model saved to: {best_path}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pneumonia classifier")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw/chest_xray",
        help="Path to chest_xray folder",
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"]
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
