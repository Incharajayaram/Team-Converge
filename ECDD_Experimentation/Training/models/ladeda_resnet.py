"""
LaDeDa-style ResNet50 for Deepfake Detection

Architecture modifications based on "Real-Time Deepfake Detection in the Real-World"
(Hebrew University of Jerusalem, 2024):

1. Replace conv1 (7x7, stride 2) with 3x3, stride 1 - preserves spatial resolution
2. Remove first maxpool layer - preserves spatial resolution
3. Output spatial patch-logit map instead of global classification
4. Use attention pooling to aggregate patch scores

NOTE: The original LaDeDa paper also replaces some 3x3 convs with 1x1 to limit
receptive field to ~9x9 pixels. This implementation uses standard ResNet50 layers
for better compatibility with pretrained weights. The receptive field is larger
but empirically works well for deepfake detection.

Spatial dimensions for 256x256 input (without maxpool):
- After conv1 (stride 1): 256x256
- After layer1: 128x128 (first block has stride 2 downsample)  
- After layer2: 64x64
- After layer3: 32x32
- After layer4: 16x16 -> 16x16 patch-logit grid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple, Optional


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over patch logits.
    
    Learns to weight patches based on their "importance" for the final decision.
    Per ECDD: attention weights must be deterministic and stable under quantization.
    
    Args:
        in_channels: Number of input feature channels (default 2048 for ResNet50)
        hidden_dim: Hidden dimension for attention network
        temperature: Temperature for softmax (higher = softer weights, more stable)
    """
    
    def __init__(self, in_channels: int = 2048, hidden_dim: int = 512, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        # Attention scoring network
        self.attention_fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        
    def forward(self, features: torch.Tensor, patch_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Feature map from backbone (B, C, H, W)
            patch_logits: Patch-level logits (B, 1, H, W)
            
        Returns:
            pooled_logit: Single image logit (B, 1)
            attention_weights: Normalized attention map (B, 1, H, W)
        """
        # Compute attention scores
        attention_scores = self.attention_fc(features)  # (B, 1, H, W)
        
        # Flatten spatial dimensions for softmax
        B, _, H, W = attention_scores.shape
        attention_flat = attention_scores.view(B, -1)  # (B, H*W)
        
        # Apply temperature-scaled softmax for stable, controllable sharpness
        attention_weights_flat = F.softmax(attention_flat / self.temperature, dim=1)  # (B, H*W)
        attention_weights = attention_weights_flat.view(B, 1, H, W)  # (B, 1, H, W)
        
        # Weighted sum of patch logits
        patch_logits_flat = patch_logits.view(B, -1)  # (B, H*W)
        pooled_logit = (patch_logits_flat * attention_weights_flat).sum(dim=1, keepdim=True)  # (B, 1)
        
        return pooled_logit, attention_weights


class LaDeDaResNet50(nn.Module):
    """
    LaDeDa-style ResNet50 for patch-based deepfake detection.
    
    Key modifications from standard ResNet50:
    - 3x3 conv1 instead of 7x7 (stride 1 instead of 2)
    - No maxpool layer after conv1
    - Patch-level 1x1 classifier instead of global fc
    - Attention pooling for image-level aggregation
    
    Output: 16x16 patch-logit map for 256x256 input.
    """
    
    def __init__(self, 
                 pretrained: bool = True,
                 freeze_layers: Optional[list] = None,
                 num_classes: int = 1):
        """
        Args:
            pretrained: Load ImageNet pretrained weights
            freeze_layers: List of layer names to freeze (e.g., ['conv1', 'layer1'])
            num_classes: Number of output classes (1 for binary)
        """
        super().__init__()
        
        # Load base ResNet50
        if pretrained:
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            base_model = resnet50(weights=None)
        
        # ===== MODIFICATION 1: Replace conv1 =====
        # Original: 7x7, stride 2, padding 3
        # LaDeDa: 3x3, stride 1, padding 1 (smaller receptive field)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Initialize with center of original 7x7 kernel if pretrained
        if pretrained:
            with torch.no_grad():
                # Take center 3x3 of original 7x7 kernel
                original_weight = base_model.conv1.weight.data  # (64, 3, 7, 7)
                center = original_weight[:, :, 2:5, 2:5]  # (64, 3, 3, 3)
                self.conv1.weight.data = center
        
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        
        # ===== MODIFICATION 2: Remove maxpool =====
        # Original has 3x3 maxpool with stride 2
        # We skip it to preserve spatial resolution
        # self.maxpool = base_model.maxpool  # REMOVED
        
        # ===== ResNet layers (with optional 1x1 modifications) =====
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # ===== MODIFICATION 3: Patch classifier head =====
        # Instead of global average pool + fc, use 1x1 conv for patch logits
        self.patch_classifier = nn.Conv2d(2048, num_classes, kernel_size=1)
        
        # ===== MODIFICATION 4: Attention pooling =====
        self.attention_pool = AttentionPooling(in_channels=2048)
        
        # Remove original fc and avgpool (not needed)
        # self.avgpool = base_model.avgpool  # REMOVED
        # self.fc = base_model.fc  # REMOVED
        
        # Freeze specified layers
        self.freeze_layers = freeze_layers or []
        self._freeze_layers()
        
    def _freeze_layers(self):
        """Freeze specified layers for finetuning efficiency."""
        freeze_map = {
            'conv1': [self.conv1, self.bn1],
            'layer1': [self.layer1],
            'layer2': [self.layer2],
            'layer3': [self.layer3],
            'layer4': [self.layer4],
        }
        
        for layer_name in self.freeze_layers:
            if layer_name in freeze_map:
                for module in freeze_map[layer_name]:
                    for param in module.parameters():
                        param.requires_grad = False
                print(f"Frozen: {layer_name}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 256, 256)
            
        Returns:
            pooled_logit: Image-level logit (B, 1)
            patch_logits: Spatial patch logits (B, 1, H, W)
            attention_map: Attention weights (B, 1, H, W)
        """
        # Backbone forward (spatial dimensions noted for 256x256 input)
        x = self.conv1(x)        # (B, 64, 256, 256) - stride 1, no reduction
        x = self.bn1(x)
        x = self.relu(x)
        # NO maxpool - preserves 256x256
        
        x = self.layer1(x)       # (B, 256, 128, 128) - first bottleneck has stride 2
        x = self.layer2(x)       # (B, 512, 64, 64)
        x = self.layer3(x)       # (B, 1024, 32, 32)
        x = self.layer4(x)       # (B, 2048, 16, 16)
        
        features = x  # Save for attention
        
        # Patch-level logits
        patch_logits = self.patch_classifier(features)  # (B, 1, 16, 16)
        
        # Attention pooling
        pooled_logit, attention_map = self.attention_pool(features, patch_logits)
        
        return pooled_logit, patch_logits, attention_map
    
    def get_patch_logit_shape(self, input_size: int = 256) -> Tuple[int, int]:
        """Return expected patch-logit map dimensions for given input size.
        
        With maxpool removed, stride reductions are:
        - layer1: 2x (first block has stride 2)
        - layer2: 2x
        - layer3: 2x
        - layer4: 1x (no downsample)
        
        Total: 256 -> 128 -> 64 -> 32 -> 32 = 32x32 output for 256x256 input
        """
        # Without maxpool: input / 8 (not input / 16)
        output_size = input_size // 8
        return (output_size, output_size)


def create_ladeda_model(pretrained: bool = True, 
                         freeze_layers: Optional[list] = None) -> LaDeDaResNet50:
    """
    Factory function to create LaDeDa model.
    
    Recommended freeze strategies:
    - Full finetuning: freeze_layers=None or []
    - Moderate finetuning: freeze_layers=['conv1', 'layer1'] (RECOMMENDED for most cases)
    - Light finetuning: freeze_layers=['conv1', 'layer1', 'layer2']
    - Head-only: freeze_layers=['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    """
    return LaDeDaResNet50(pretrained=pretrained, freeze_layers=freeze_layers)


if __name__ == "__main__":
    # Test the model
    print("Testing LaDeDa ResNet50...")
    
    model = create_ladeda_model(pretrained=True, freeze_layers=['conv1', 'layer1'])
    model.eval()
    
    # Test input
    x = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        pooled, patches, attention = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Pooled logit shape: {pooled.shape}")
    print(f"Patch logits shape: {patches.shape}")
    print(f"Attention map shape: {attention.shape}")
    print(f"Expected patch grid: {model.get_patch_logit_shape(256)}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
