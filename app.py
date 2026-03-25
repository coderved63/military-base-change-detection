"""Gradio web demo for satellite change detection.

Upload before/after satellite image pairs, select a model and checkpoint, and
view the predicted change mask, overlay, and change-area statistics.

Defaults (model, checkpoint, port, share) are read from the ``gradio`` section
of ``configs/config.yaml``.

Usage:
    python app.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import yaml

from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from inference import load_and_preprocess, sliding_window_inference
from models import get_model
from utils.visualization import overlay_changes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Globals (model cache to avoid reloading on every prediction)
# ---------------------------------------------------------------------------

_cached_model: Optional[torch.nn.Module] = None
_cached_model_key: Optional[str] = None  # "model_name::checkpoint_path"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_config: Optional[Dict[str, Any]] = None


def _load_config() -> Dict[str, Any]:
    """Load and cache the project config.

    Returns:
        Full config dict.
    """
    global _config
    if _config is None:
        config_path = Path("configs/config.yaml")
        with open(config_path, "r") as fh:
            _config = yaml.safe_load(fh)
    return _config


def _load_model(model_name: str, checkpoint_path: str) -> torch.nn.Module:
    """Load a model, re-using the cache if name + checkpoint match.

    Args:
        model_name: Architecture name (``siamese_cnn``, ``unet_pp``, ``changeformer``).
        checkpoint_path: Path to the ``.pth`` checkpoint file.

    Returns:
        Model in eval mode on the current device.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    global _cached_model, _cached_model_key

    cache_key = f"{model_name}::{checkpoint_path}"
    if _cached_model is not None and _cached_model_key == cache_key:
        return _cached_model

    config = _load_config()
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = get_model(model_name, config).to(_device)
    ckpt = torch.load(ckpt_path, map_location=_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _cached_model = model
    _cached_model_key = cache_key
    logger.info("Loaded model %s from %s", model_name, checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# Preprocessing helper (numpy RGB uint8 → tensor)
# ---------------------------------------------------------------------------

def _numpy_to_tensor(
    img: np.ndarray,
    patch_size: int = 256,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Convert a uint8 RGB numpy image to a normalised, padded tensor.

    Args:
        img: Input image ``[H, W, 3]``, uint8, RGB.
        patch_size: Pad to a multiple of this value.

    Returns:
        Tuple of ``(tensor [1, 3, H_pad, W_pad], (orig_h, orig_w))``.
    """
    orig_h, orig_w = img.shape[:2]

    pad_h = (patch_size - orig_h % patch_size) % patch_size
    pad_w = (patch_size - orig_w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    img_f = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_f = (img_f - mean) / std

    tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, (orig_h, orig_w)


# ---------------------------------------------------------------------------
# Prediction function (called by Gradio)
# ---------------------------------------------------------------------------

def predict(
    before_image: Optional[np.ndarray],
    after_image: Optional[np.ndarray],
    model_name: str,
    checkpoint_path: str,
    threshold: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Run change detection and return visualisations + summary text.

    Args:
        before_image: Before image as numpy ``[H, W, 3]`` RGB uint8.
        after_image: After image as numpy ``[H, W, 3]`` RGB uint8.
        model_name: Architecture name.
        checkpoint_path: Path to checkpoint file.
        threshold: Binarisation threshold for predictions.

    Returns:
        Tuple of ``(change_mask, overlay_image, summary_text)``.
        - ``change_mask``: uint8 grayscale ``[H, W]`` (0 or 255).
        - ``overlay_image``: uint8 RGB ``[H, W, 3]``.
        - ``summary_text``: Markdown string with change statistics.
    """
    if before_image is None or after_image is None:
        return None, None, "Please upload both before and after images."

    config = _load_config()
    patch_size: int = config.get("dataset", {}).get("patch_size", 256)

    # Load model
    try:
        model = _load_model(model_name, checkpoint_path)
    except FileNotFoundError as exc:
        return None, None, f"Error: {exc}"

    # Preprocess
    tensor_a, (orig_h, orig_w) = _numpy_to_tensor(before_image, patch_size)
    tensor_b, _ = _numpy_to_tensor(after_image, patch_size)

    # Tiled inference
    prob_map = sliding_window_inference(model, tensor_a, tensor_b, patch_size, _device)
    prob_map = prob_map[:, :, :orig_h, :orig_w]
    prob_np = prob_map.squeeze().numpy()  # [H, W]

    # Binary change mask
    binary_mask = (prob_np > threshold).astype(np.uint8) * 255

    # Overlay on after image
    pred_tensor = (prob_map.squeeze(0) >= threshold).float()  # [1, H, W]
    img_b_tensor = tensor_b.squeeze()[:, :orig_h, :orig_w]    # [3, H, W]
    overlay_rgb = overlay_changes(
        img_after=img_b_tensor,
        mask_pred=pred_tensor,
        alpha=0.4,
        color=(255, 0, 0),
    )

    # Change statistics
    total_pixels = orig_h * orig_w
    changed_pixels = int(binary_mask.sum() // 255)
    pct_changed = (changed_pixels / total_pixels) * 100.0

    summary = (
        f"### Change Detection Summary\n"
        f"- **Image size**: {orig_w} x {orig_h}\n"
        f"- **Total pixels**: {total_pixels:,}\n"
        f"- **Changed pixels**: {changed_pixels:,}\n"
        f"- **Area changed**: {pct_changed:.2f}%\n"
        f"- **Model**: {model_name}\n"
        f"- **Threshold**: {threshold}"
    )

    return binary_mask, overlay_rgb, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Construct the Gradio Blocks interface.

    Returns:
        A ``gr.Blocks`` application ready to ``.launch()``.
    """
    config = _load_config()
    gradio_cfg = config.get("gradio", {})

    with gr.Blocks(
        title="Military Base Change Detection",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# Military Base Change Detection\n"
            "Upload **before** and **after** satellite images to detect "
            "construction, infrastructure changes, and runway development."
        )

        # ---- Inputs ---------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                before_img = gr.Image(
                    label="Before Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
            with gr.Column(scale=1):
                after_img = gr.Image(
                    label="After Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )

        # ---- Controls -------------------------------------------------
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["siamese_cnn", "unet_pp", "changeformer"],
                value=gradio_cfg.get("default_model", "unet_pp"),
                label="Model Architecture",
            )
            checkpoint_input = gr.Textbox(
                value=gradio_cfg.get("default_checkpoint", "checkpoints/unet_pp_best.pth"),
                label="Checkpoint Path",
            )
            threshold_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Detection Threshold",
            )

        detect_btn = gr.Button("Detect Changes", variant="primary", size="lg")

        # ---- Outputs --------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                change_mask_out = gr.Image(label="Change Mask")
            with gr.Column(scale=1):
                overlay_out = gr.Image(label="Overlay (changes in red)")

        summary_out = gr.Markdown(label="Summary")

        # ---- Wiring ---------------------------------------------------
        detect_btn.click(
            fn=predict,
            inputs=[
                before_img,
                after_img,
                model_dropdown,
                checkpoint_input,
                threshold_slider,
            ],
            outputs=[change_mask_out, overlay_out, summary_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Gradio demo server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = _load_config()
    gradio_cfg = config.get("gradio", {})

    demo = build_demo()
    demo.launch(
        server_port=gradio_cfg.get("server_port", 7860),
        share=gradio_cfg.get("share", False),
    )


if __name__ == "__main__":
    main()
