"""
Optional Gradio demo for pneumonia detection with Grad-CAM / Eigen-CAM overlays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cam import generate_eigencam, generate_gradcam, overlay_heatmap
from src.dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_resnet50.pth"

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

_model = None
_backbone = "resnet50"


def load_model(checkpoint_path: str = str(DEFAULT_CHECKPOINT)):
    global _model, _backbone
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nTrain the model first: python -m src.train"
        )
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    _backbone = ckpt.get("backbone", "resnet50")
    _model = build_model(backbone=_backbone, num_classes=2, pretrained=False)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.to(DEVICE).eval()


def predict(image: Image.Image):
    if _model is None:
        load_model()

    img_rgb = image.convert("RGB").resize((224, 224))
    tensor = _transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    rgb = np.array(img_rgb).astype(np.float32) / 255.0
    gradcam = generate_gradcam(_model, tensor, pred_idx, _backbone)
    eigencam = generate_eigencam(_model, tensor, _backbone)

    grad_img = overlay_heatmap(rgb, gradcam)
    eigen_img = overlay_heatmap(rgb, eigencam)

    label = f"**{pred_class}** ({confidence:.1%} confidence)\n\n"
    label += f"NORMAL: {probs[0]:.1%} | PNEUMONIA: {probs[1]:.1%}"

    return label, grad_img, eigen_img


def build_demo():
    try:
        load_model()
    except FileNotFoundError as e:
        print(e)

    with gr.Blocks(title="Chest X-Ray Pneumonia Detection") as demo:
        gr.Markdown(
            "# Chest X-Ray Pneumonia Detection\n"
            "Upload a chest X-ray to classify **Normal** vs **Pneumonia** "
            "with **Grad-CAM** and **Eigen-CAM** explanations.\n\n"
            "*For research/education only — not for clinical use.*"
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Chest X-Ray")
            with gr.Column():
                out_label = gr.Markdown(label="Prediction")
                grad_out = gr.Image(label="Grad-CAM")
                eigen_out = gr.Image(label="Eigen-CAM")

        btn = gr.Button("Analyze", variant="primary")
        btn.click(predict, inputs=inp, outputs=[out_label, grad_out, eigen_out])
        gr.Examples(
            examples=[],
            inputs=inp,
            label="Examples (add paths after training)",
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
