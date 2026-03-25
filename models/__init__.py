"""Model factory for change detection models.

Provides a unified interface to instantiate any supported model by name.
"""

from typing import Any, Dict

import torch.nn as nn

from .changeformer import ChangeFormer
from .siamese_cnn import SiameseCNN
from .unet_pp import UNetPPChangeDetection

_MODEL_REGISTRY: Dict[str, type] = {
    "siamese_cnn": SiameseCNN,
    "unet_pp": UNetPPChangeDetection,
    "changeformer": ChangeFormer,
}


def get_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """Instantiate a change detection model by name.

    Args:
        model_name: One of 'siamese_cnn', 'unet_pp', 'changeformer'.
        config: Full config dict; model-specific section is extracted internally.

    Returns:
        Initialized model (nn.Module).

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_MODEL_REGISTRY.keys())}")

    model_cls = _MODEL_REGISTRY[model_name]
    model_config = config.get(model_name, {})
    return model_cls(**model_config)
