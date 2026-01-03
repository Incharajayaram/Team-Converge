"""Tiny LaDeDa - lightweight edge student model (1,297 parameters)."""

import torch
import torch.nn as nn
from pathlib import Path
from models.reference.Tiny_LaDeDa import tiny_ladeda


class TinyLaDeDa(nn.Module):
    """
    Ultra-lightweight student model for edge deployment.

    - Architecture: Single layer, 8 channels, 1,297 total parameters
    - Input: 256x256 RGB images (normalized with ImageNet mean/std)
    - Output: (B, 1, 126, 126) patch-logit map when pool=False
    - Preprocessing: Gradient-based (right_diag diagonal kernel)
    - Pretrained weights: ForenSynth_Tiny_LaDeDa.pth available

    This is the exact reference implementation, not a configurable variant.
    """

    def __init__(self, pretrained=False, pretrained_path="weights/student/ForenSynth_Tiny_LaDeDa.pth"):
        """
        Args:
            pretrained: Load pretrained weights
            pretrained_path: Path to pretrained weights
        """
        super().__init__()
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path

        # Initialize reference tiny_ladeda with gradient preprocessing
        # Output patch-logit maps (pool=False)
        self.model = tiny_ladeda(
            preprocess_type="right_diag",  # Gradient-based preprocessing
            num_classes=1,
            pool=False  # Critical: return spatial patch-logit maps
        )

        # Ensure float32 dtype
        self.model = self.model.float()

        # Load pretrained weights if available
        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load pretrained weights from disk."""
        weight_path = Path(self.pretrained_path)

        # Convert to absolute path if relative
        if not weight_path.is_absolute():
            # Assume relative to project root
            project_root = Path(__file__).parent.parent.parent  # models/student/ -> deepfake-patch-audit/
            weight_path = project_root / weight_path

        if weight_path.exists():
            try:
                state_dict = torch.load(str(weight_path), map_location='cpu')
                self.model.load_state_dict(state_dict)
                print(f"✓ Loaded student weights from {weight_path}")
            except Exception as e:
                print(f"⚠ Failed to load weights from {weight_path}: {e}")
                print("  Training with random initialization")
        else:
            print(f"⚠ Weights not found at {weight_path}")
            print("  Training with random initialization")

    def forward(self, x):
        """
        Forward pass - returns patch-logit map.

        Args:
            x: Input tensor (B, 3, 256, 256), normalized with ImageNet mean/std

        Returns:
            Patch-logit map (B, 1, 126, 126) - one logit per spatial location
        """
        x = x.float()
        return self.model(x)

    def get_patch_logits(self, x):
        """
        Alias for forward() - explicitly returns patch-logit map.

        Args:
            x: Input tensor (B, 3, 256, 256)

        Returns:
            Patch-logit map (B, 1, 126, 126)
        """
        return self.forward(x)

    def get_patch_probabilities(self, x):
        """
        Get patch-wise fake probabilities.

        Args:
            x: Input tensor (B, 3, 256, 256)

        Returns:
            Patch probabilities (B, 1, 126, 126) after sigmoid
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
