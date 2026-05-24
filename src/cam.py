"""
Grad-CAM and Eigen-CAM explainability for chest X-ray predictions.

Uses pytorch-grad-cam for Grad-CAM and a custom Eigen-CAM implementation
(principal component of activation maps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CLASS_NAMES, IDX_TO_CLASS, denormalize
from .model import BackboneName, get_target_layer


def _tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor (C,H,W) to RGB float [0,1] for overlay."""
    img = denormalize(tensor).clamp(0, 1).permute(1, 2, 0).numpy()
    return img.astype(np.float32)


class EigenCAM:
    """
    Eigen-CAM: projects activations onto the first principal component.

    Often produces cleaner, less noisy heatmaps than Grad-CAM.
    """

    def __init__(self, model: nn.Module, target_layers: list) -> None:
        self.model = model
        self.target_layers = target_layers
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        for layer in target_layers:
            self._hooks.append(
                layer.register_forward_hook(self._save_activation)
            )

    def _save_activation(self, module, input, output) -> None:
        self.activations = output

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: Optional[list] = None,
    ) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        if self.activations is None:
            raise RuntimeError("No activations captured")

        activations = self.activations.detach().cpu().numpy()
        # activations: (B, C, H, W)
        b, c, h, w = activations.shape
        cam = np.zeros((b, h, w), dtype=np.float32)

        for i in range(b):
            act = activations[i].reshape(c, -1)  # (C, H*W)
            # Center activations
            act = act - act.mean(axis=1, keepdims=True)
            try:
                u, s, vt = np.linalg.svd(act, full_matrices=False)
                cam[i] = (s[0] * vt[0]).reshape(h, w)
            except np.linalg.LinAlgError:
                cam[i] = act.mean(axis=0).reshape(h, w)

            cam[i] = np.maximum(cam[i], 0)
            cam[i] = (cam[i] - cam[i].min()) / (cam[i].max() - cam[i].min() + 1e-8)

        return cam

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    backbone: BackboneName,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for a single image."""
    target_layer = get_target_layer(model, backbone)
    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )
    return grayscale_cam[0]


def generate_eigencam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    backbone: BackboneName,
) -> np.ndarray:
    """Generate Eigen-CAM heatmap for a single image."""
    target_layer = get_target_layer(model, backbone)
    eigen_cam = EigenCAM(model=model, target_layers=[target_layer])
    cam = eigen_cam(input_tensor)
    eigen_cam.remove_hooks()
    return cam[0]


def overlay_heatmap(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay heatmap on RGB image using grad-cam utilities."""
    return show_cam_on_image(rgb_image, heatmap, use_rgb=True, image_weight=alpha)


def plot_cam_comparison(
    rgb_image: np.ndarray,
    gradcam: np.ndarray,
    eigencam: np.ndarray,
    true_label: str,
    pred_label: str,
    confidence: float,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create side-by-side: Original | Grad-CAM | Eigen-CAM."""
    grad_overlay = overlay_heatmap(rgb_image, gradcam)
    eigen_overlay = overlay_heatmap(rgb_image, eigencam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [rgb_image, grad_overlay, eigen_overlay]
    titles = ["Original X-Ray", "Grad-CAM", "Eigen-CAM"]

    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=12)
        ax.axis("off")

    status = "✓" if true_label == pred_label else "✗"
    fig.suptitle(
        f"{title or ''} True: {true_label} | Pred: {pred_label} "
        f"({confidence:.1%}) {status}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


@torch.no_grad()
def generate_cam_comparison(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    backbone: BackboneName,
    output_dir: Path,
    num_samples: int = 12,
) -> None:
    """Generate Grad-CAM vs Eigen-CAM comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    count = 0

    for images, labels, paths in tqdm(loader, desc="Generating CAMs"):
        for i in range(images.size(0)):
            if count >= num_samples:
                return

            img_tensor = images[i : i + 1].to(device)
            label = labels[i].item()
            true_name = IDX_TO_CLASS[label]

            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class = outputs.argmax(dim=1).item()
            pred_name = IDX_TO_CLASS[pred_class]
            confidence = probs[pred_class].item()

            rgb = _tensor_to_rgb_uint8(images[i])
            gradcam = generate_gradcam(model, img_tensor, pred_class, backbone)
            eigencam = generate_eigencam(model, img_tensor, backbone)

            fname = f"cam_{count:02d}_{true_name}_pred_{pred_name}.png"
            plot_cam_comparison(
                rgb,
                gradcam,
                eigencam,
                true_name,
                pred_name,
                confidence,
                save_path=output_dir / fname,
            )
            count += 1


def save_failure_cases(
    model: nn.Module,
    labels: np.ndarray,
    preds: np.ndarray,
    paths: list[str],
    loader: DataLoader,
    device: torch.device,
    backbone: BackboneName,
    output_dir: Path,
    num_cases: int = 6,
) -> None:
    """
    Save heatmaps for worst failure cases:
    - False negatives (missed pneumonia)
    - False positives (false alarm)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pneumonia_idx = 1
    false_negatives = np.where((labels == pneumonia_idx) & (preds != pneumonia_idx))[0]
    false_positives = np.where((labels != pneumonia_idx) & (preds == pneumonia_idx))[0]

    cases = []
    for idx in false_negatives[: num_cases // 2]:
        cases.append((idx, "false_negative"))
    for idx in false_positives[: num_cases // 2]:
        cases.append((idx, "false_positive"))

    path_to_tensor = {}
    for images, batch_labels, batch_paths in loader:
        for j, p in enumerate(batch_paths):
            path_to_tensor[p] = images[j]

    model.eval()
    for case_idx, (data_idx, case_type) in enumerate(cases):
        img_path = paths[data_idx]
        if img_path not in path_to_tensor:
            continue

        tensor = path_to_tensor[img_path].unsqueeze(0).to(device)
        label = labels[data_idx]
        pred = preds[data_idx]
        true_name = IDX_TO_CLASS[label]
        pred_name = IDX_TO_CLASS[pred]

        outputs = model(tensor)
        confidence = torch.softmax(outputs, dim=1)[0, pred].item()

        rgb = _tensor_to_rgb_uint8(path_to_tensor[img_path])
        gradcam = generate_gradcam(model, tensor, int(pred), backbone)
        eigencam = generate_eigencam(model, tensor, backbone)

        fname = f"{case_type}_{case_idx}_{true_name}_pred_{pred_name}.png"
        plot_cam_comparison(
            rgb,
            gradcam,
            eigencam,
            true_name,
            pred_name,
            confidence,
            save_path=output_dir / fname,
            title=f"[{case_type.replace('_', ' ').title()}]",
        )
