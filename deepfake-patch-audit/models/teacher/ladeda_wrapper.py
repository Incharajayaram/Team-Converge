"""Wrapper for pretrained LaDeDa9 teacher model."""

import torch
import torch.nn as nn
from pathlib import Path
from models.reference.LaDeDa import LaDeDa9


class LaDeDaWrapper(nn.Module):
    """
    Wrapper for pretrained LaDeDa9 teacher model.

    - Input: 256x256 RGB images (normalized with ImageNet mean/std)
    - Output: (B, 1, 31, 31) patch-logit map when pool=False
    - Preprocessing: NPR (Nearest Neighbor Residual)
    - Pretrained weights: WildRF_LaDeDa.pth or similar
    """

    def __init__(self, pretrained=True, pretrained_path="weights/teacher/WildRF_LaDeDa.pth", freeze_backbone=True):
        """
        Args:
            pretrained: Load pretrained weights
            pretrained_path: Path to pretrained weights (relative or absolute)
            freeze_backbone: Freeze all parameters during training
        """
        super().__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.pretrained_path = pretrained_path

        # Initialize LaDeDa9 model with NPR preprocessing
        # Output patch-logit maps (pool=False)
        self.model = LaDeDa9(
            preprocess_type="NPR",
            num_classes=1,
            pool=False  # Critical: return spatial patch-logit maps
        )

        # Ensure float32 dtype
        self.model = self.model.float()

        # Load pretrained weights if available
        if pretrained:
            self._load_pretrained_weights()

        # Freeze if requested
        if freeze_backbone:
            self._freeze_parameters()

    def _load_pretrained_weights(self):
        """Load pretrained weights from disk."""
        weight_path = Path(self.pretrained_path)

        # Convert to absolute path if relative
        if not weight_path.is_absolute():
            # Assume relative to project root
            project_root = Path(__file__).parent.parent.parent  # models/teacher/ -> deepfake-patch-audit/
            weight_path = project_root / weight_path

        if weight_path.exists():
            try:
                state_dict = torch.load(str(weight_path), map_location='cpu')
                self.model.load_state_dict(state_dict)
                print(f"✓ Loaded teacher weights from {weight_path}")
            except Exception as e:
                print(f"⚠ Failed to load weights from {weight_path}: {e}")
                print("  Training with random initialization")
        else:
            print(f"⚠ Weights not found at {weight_path}")
            print("  Training with random initialization")

    def _freeze_parameters(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass - returns patch-logit map.

        Args:
            x: Input tensor (B, 3, 256, 256), normalized with ImageNet mean/std

        Returns:
            Patch-logit map (B, 1, 31, 31) - one logit per spatial location
        """
        x = x.float()
        return self.model(x)

    def get_patch_logits(self, x):
        """
        Alias for forward() - explicitly returns patch-logit map.

        Args:
            x: Input tensor (B, 3, 256, 256)

        Returns:
            Patch-logit map (B, 1, 31, 31)
        """
        return self.forward(x)

    def get_patch_probabilities(self, x):
        """
        Get patch-wise fake probabilities.

        Args:
            x: Input tensor (B, 3, 256, 256)

        Returns:
            Patch probabilities (B, 1, 31, 31) after sigmoid
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
