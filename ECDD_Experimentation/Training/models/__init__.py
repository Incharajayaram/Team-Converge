"""ECDD Training Models Package.

Contains model architectures for deepfake detection.
"""

from .ladeda_resnet import LaDeDaResNet50, AttentionPooling, create_ladeda_model

__all__ = [
    "LaDeDaResNet50",
    "AttentionPooling", 
    "create_ladeda_model",
]
