"""
CNN models with transfer learning: ResNet50 (primary) and EfficientNet-B0.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:
    timm = None

BackboneName = Literal["resnet50", "efficientnet_b0"]


def build_model(
    backbone: BackboneName = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = True,
    unfreeze_from_layer: Optional[int] = None,
) -> nn.Module:
    """
    Build a transfer-learning classifier.

    Args:
        backbone: 'resnet50' or 'efficientnet_b0'
        num_classes: Number of output classes (2 for Normal/Pneumonia)
        pretrained: Use ImageNet weights
        dropout: Dropout before final linear layer
        freeze_backbone: Freeze all backbone layers initially
        unfreeze_from_layer: If set, unfreeze layers from this index onward
    """
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "efficientnet_b0":
        if timm is None:
            raise ImportError("timm is required for EfficientNet. pip install timm")
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if freeze_backbone:
        _freeze_backbone(model, backbone)

    if unfreeze_from_layer is not None:
        _unfreeze_from_layer(model, backbone, unfreeze_from_layer)

    return model


def _freeze_backbone(model: nn.Module, backbone: BackboneName) -> None:
    if backbone == "resnet50":
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    elif backbone == "efficientnet_b0":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False


def _unfreeze_from_layer(
    model: nn.Module, backbone: BackboneName, layer_idx: int
) -> None:
    """Progressively unfreeze later layers for fine-tuning."""
    if backbone == "resnet50":
        layers = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.fc,
        ]
        for i, layer in enumerate(layers):
            if i >= layer_idx:
                for param in layer.parameters():
                    param.requires_grad = True
    elif backbone == "efficientnet_b0":
        blocks = list(model.blocks.children())
        for i, block in enumerate(blocks):
            if i >= layer_idx:
                for param in block.parameters():
                    param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True


def get_target_layer(model: nn.Module, backbone: BackboneName) -> nn.Module:
    """Return the convolutional layer used for CAM visualization."""
    if backbone == "resnet50":
        return model.layer4[-1]
    if backbone == "efficientnet_b0":
        return model.conv_head
    raise ValueError(f"Unknown backbone: {backbone}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def unfreeze_layers(
    model: nn.Module, backbone: BackboneName, layer_idx: int
) -> None:
    """Unfreeze backbone layers from layer_idx onward (for fine-tuning)."""
    _unfreeze_from_layer(model, backbone, layer_idx)
