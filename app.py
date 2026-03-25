"""Gradio web demo for change detection inference.

Provides an interactive interface to upload before/after satellite image pairs
and visualize predicted change masks with overlays.

Usage:
    python app.py
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import yaml

from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from inference import preprocess_image, sliding_window_inference
from models import get_model
from utils.visualization import denormalize, overlay_changes

logger = logging.getLogger(__name__)

# Global model cache
_model: Optional[torch.nn.Module] = None
_model_name: Optional[str] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_config = None


def load_config() -> dict:
    """Load project config from YAML.

    Returns:
        Config dictionary.
    """
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_name: str, checkpoint_path: str) -> torch.nn.Module:
    """Load a change detection model with caching.

    Args:
        model_name: Name of the model architecture.
        checkpoint_path: Path to the model checkpoint.

    Returns:
        Loaded model in eval mode.
    """
    global _model, _model_name, _config

    if _config is None:
        _config = load_config()

    if _model is not None and _model_name == model_name:
        return _model

    model = get_model(model_name, _config).to(_device)
    ckpt = torch.load(checkpoint_path, map_location=_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model = model
    _model_name = model_name
    logger.info("Loaded model: %s from %s", model_name, checkpoint_path)
    return model


def predict(
    before_image: np.ndarray,
    after_image: np.ndarray,
    model_name: str,
    checkpoint_path: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run change detection on a pair of images.

    Args:
        before_image: Before image as numpy array (RGB, uint8).
        after_image: After image as numpy array (RGB, uint8).
        model_name: Model architecture name.
        checkpoint_path: Path to model weights.
        threshold: Binarization threshold.

    Returns:
        Tuple of (binary change mask, overlay visualization).
    """
    model = load_model(model_name, checkpoint_path)
    patch_size = 256

    # Preprocess both images
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        img_f = img.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img_f = (img_f - mean) / std
        return torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).float()

    orig_h, orig_w = before_image.shape[:2]
    tensor_a = _to_tensor(before_image)
    tensor_b = _to_tensor(after_image)

    # Run inference
    prob_map = sliding_window_inference(model, tensor_a, tensor_b, patch_size, _device)
    prob_map = prob_map[:, :, :orig_h, :orig_w]

    # Binary mask
    mask_np = prob_map.squeeze().numpy()
    binary_mask = (mask_np > threshold).astype(np.uint8) * 255

    # Overlay on after image
    overlay = after_image.copy().astype(np.float32) / 255.0
    change_pixels = mask_np > threshold
    overlay[change_pixels, 0] = np.clip(overlay[change_pixels, 0] * 0.6 + 0.4, 0, 1)
    overlay[change_pixels, 1] = overlay[change_pixels, 1] * 0.6
    overlay[change_pixels, 2] = overlay[change_pixels, 2] * 0.6
    overlay = (overlay * 255).astype(np.uint8)

    return binary_mask, overlay


def build_demo() -> gr.Blocks:
    """Build the Gradio demo interface.

    Returns:
        Gradio Blocks application.
    """
    config = load_config()
    gradio_cfg = config.get("gradio", {})

    with gr.Blocks(title="Military Base Change Detection") as demo:
        gr.Markdown("# Military Base Change Detection")
        gr.Markdown("Upload before/after satellite image pairs to detect construction and infrastructure changes.")

        with gr.Row():
            with gr.Column():
                before_img = gr.Image(label="Before Image", type="numpy")
                after_img = gr.Image(label="After Image", type="numpy")
            with gr.Column():
                change_mask = gr.Image(label="Change Mask")
                overlay_img = gr.Image(label="Overlay")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["siamese_cnn", "unet_pp", "changeformer"],
                value=gradio_cfg.get("default_model", "unet_pp"),
                label="Model",
            )
            checkpoint_input = gr.Textbox(
                value=gradio_cfg.get("default_checkpoint", "checkpoints/unet_pp_best.pth"),
                label="Checkpoint Path",
            )
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Detection Threshold",
            )

        detect_btn = gr.Button("Detect Changes", variant="primary")
        detect_btn.click(
            fn=predict,
            inputs=[before_img, after_img, model_dropdown, checkpoint_input, threshold_slider],
            outputs=[change_mask, overlay_img],
        )

    return demo


def main() -> None:
    """Launch the Gradio demo."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config()
    gradio_cfg = config.get("gradio", {})

    demo = build_demo()
    demo.launch(
        server_port=gradio_cfg.get("server_port", 7860),
        share=gradio_cfg.get("share", False),
    )


if __name__ == "__main__":
    main()
