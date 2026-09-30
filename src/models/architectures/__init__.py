"""
Model registry and factory.

Maps model name strings (from YAML config) to their corresponding
PyTorch ``nn.Module`` classes. Use ``build_model(config)`` to instantiate.
"""

from typing import Dict, Type
import torch.nn as nn

from src.models.architectures.cnn import CNN
from src.models.architectures.rnn import RNNModel
from src.models.architectures.lstm import LSTMModel
from src.models.architectures.lstm_attention import LSTMAttentionModel
from src.models.architectures.bilstm import BiLSTMModel
from src.models.architectures.bilstm_attention import BiLSTMAttentionModel
from src.models.architectures.residual_bilstm import ResidualBiLSTMModel
from src.models.architectures.cnn_lstm import CNNLSTMModel
from src.models.architectures.cnn_bilstm import CNNBiLSTMModel
from src.models.architectures.vision_transformer import VisionTransformer
from src.models.architectures.swin_transformer import SwinTransformer

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn": CNN,
    "rnn": RNNModel,
    "lstm": LSTMModel,
    "lstm_attention": LSTMAttentionModel,
    "bilstm": BiLSTMModel,
    "bilstm_attention": BiLSTMAttentionModel,
    "residual_bilstm": ResidualBiLSTMModel,
    "cnn_lstm": CNNLSTMModel,
    "cnn_bilstm": CNNBiLSTMModel,
    "vision_transformer": VisionTransformer,
    "swin_transformer": SwinTransformer,
}


def get_available_models():
    """Return list of registered model names."""
    return list(MODEL_REGISTRY.keys())


def build_model(config: Dict) -> nn.Module:
    """
    Instantiate a model from config.

    The ``config['model']['name']`` key selects the architecture.
    Model-specific hyper-parameters are read from
    ``config['model'][<model_name>]``.

    Args:
        config: Full project configuration dictionary.

    Returns:
        An ``nn.Module`` instance ready for training.

    Raises:
        ValueError: If the model name is not in the registry.
    """
    model_name = config["model"]["name"].lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {get_available_models()}"
        )

    data_cfg = config["data"]
    model_params = config["model"].get(model_name, {})

    # common parameters every model receives
    common = {
        "num_classes": data_cfg.get("num_classes", 7),
        "image_size": tuple(data_cfg.get("image_size", [48, 48])),
        "in_channels": 1 if data_cfg.get("color_mode", "grayscale") == "grayscale" else 3,
    }

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**common, **model_params)
    return model
