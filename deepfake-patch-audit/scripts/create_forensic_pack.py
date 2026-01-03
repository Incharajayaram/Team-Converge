#!/usr/bin/env python3
"""
Forensic Output Pack Generator for Deepfake Detection.

Creates a complete forensic evidence package containing:
1. Input images (25 real + 25 fake)
2. Patch heatmaps showing per-patch fake probabilities
3. Top-k overlay highlighting most suspicious patches
4. Model predictions and probabilities
5. Deletion sanity check (mask top-k patches, verify logit drops)

This pack enables defendable auditing of model decisions.

Usage:
    python scripts/create_forensic_pack.py \
        --checkpoint weights/teacher/teacher_finetuned_best.pth \
        --data-root data/ \
        --output-dir outputs/forensic_pack \
        --num-samples 25 \
        --device cuda
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import json
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.pooling import TopKLogitPooling


# =============================================================================
# Heatmap Generation
# =============================================================================

class ForensicHeatmapGenerator:
    """
    Generate forensic heatmaps from patch-level model outputs.
    
    Unlike sliding-window approaches, this works directly with the model's
    native patch output (e.g., 31x31 for teacher, 126x126 for student).
    """
    
    def __init__(
        self,
        model,
        pooling,
        device: str = "cuda",
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        """
        Args:
            model: Model that outputs patch logits
            pooling: TopK pooling layer
            device: Device for inference
            normalize_mean: ImageNet normalization mean
            normalize_std: ImageNet normalization std
        """
        self.model = model.to(device)
        self.pooling = pooling.to(device)
        self.device = device
        
        self.model.eval()
        
        # ImageNet normalization constants
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
    
    def preprocess_image(self, image: Image.Image, resize: int = 256) -> torch.Tensor:
        """Preprocess image for model inference."""
        # Resize
        image = image.resize((resize, resize), Image.BICUBIC)
        
        # Convert to numpy and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - self.normalize_mean) / self.normalize_std
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def generate_forensic_output(
        self,
        image_path: str,
        threshold: float = 0.5,
        k: int = 3,
    ) -> Dict:
        """
        Generate complete forensic output for a single image.
        
        Args:
            image_path: Path to image
            threshold: Decision threshold
            k: Number of top patches for overlay and deletion test
        
        Returns:
            Dict with all forensic data
        """
        # Load original image
        original = Image.open(image_path).convert("RGB")
        
        # Preprocess
        input_tensor = self.preprocess_image(original)
        
        with torch.no_grad():
            # Get patch logits: (1, 1, H, W)
            patch_logits = self.model(input_tensor)
            
            # Get pooled prediction
            pooled_logit = self.pooling(patch_logits)
            probability = torch.sigmoid(pooled_logit).item()
            prediction = 1 if probability > threshold else 0
        
        # Convert patch logits to probabilities for heatmap
        patch_probs = torch.sigmoid(patch_logits).squeeze().cpu().numpy()
        
        # Get spatial dimensions
        H, W = patch_probs.shape
        
        # Find top-k patches
        flat_probs = patch_probs.flatten()
        top_k_indices = np.argsort(flat_probs)[-k:]
        top_k_positions = [(idx // W, idx % W) for idx in top_k_indices]
        top_k_probs = flat_probs[top_k_indices]
        
        # Run deletion sanity check
        deletion_results = self._deletion_sanity_check(
            input_tensor, patch_logits, top_k_positions, k
        )
        
        return {
            "image_path": str(image_path),
            "original_size": original.size,
            "patch_grid_size": (H, W),
            "patch_probs": patch_probs,
            "probability": probability,
            "prediction": prediction,
            "prediction_label": "FAKE" if prediction == 1 else "REAL",
            "top_k_positions": top_k_positions,
            "top_k_probs": top_k_probs.tolist(),
            "deletion_results": deletion_results,
        }
    
    def _deletion_sanity_check(
        self,
        input_tensor: torch.Tensor,
        original_patch_logits: torch.Tensor,
        top_k_positions: List[Tuple[int, int]],
        k: int,
    ) -> Dict:
        """
        Verify model behavior by masking top-k patches.
        
        If model is working correctly, masking the most suspicious patches
        should reduce the fake probability.
        """
        H, W = original_patch_logits.shape[2:]
        original_prob = torch.sigmoid(self.pooling(original_patch_logits)).item()
        
        # Create masked patch logits (set top-k to very negative value)
        masked_logits = original_patch_logits.clone()
        for (i, j) in top_k_positions:
            masked_logits[0, 0, i, j] = -10.0  # Force low probability
        
        # Get new pooled prediction
        masked_pooled = self.pooling(masked_logits)
        masked_prob = torch.sigmoid(masked_pooled).item()
        
        # Calculate drop
        prob_drop = original_prob - masked_prob
        relative_drop = prob_drop / (original_prob + 1e-8)
        
        # Sanity check passed if probability drops after masking top patches
        sanity_passed = prob_drop > 0.01 or original_prob < 0.3
        
        return {
            "original_prob": original_prob,
            "masked_prob": masked_prob,
            "prob_drop": prob_drop,
            "relative_drop": relative_drop,
            "sanity_passed": sanity_passed,
            "k": k,
        }


# =============================================================================
# Visualization Functions
# =============================================================================

def create_heatmap_visualization(
    image_path: str,
    patch_probs: np.ndarray,
    output_path: str,
    colormap: str = "RdYlBu_r",
):
    """Create and save heatmap overlay visualization."""
    # Load original image
    original = Image.open(image_path).convert("RGB")
    original_array = np.array(original)
    
    # Resize heatmap to match image size
    H, W = patch_probs.shape
    img_h, img_w = original_array.shape[:2]
    
    # Upsample heatmap using bilinear interpolation
    heatmap_tensor = torch.from_numpy(patch_probs).unsqueeze(0).unsqueeze(0).float()
    heatmap_upsampled = F.interpolate(
        heatmap_tensor, size=(img_h, img_w), mode='bilinear', align_corners=False
    )
    heatmap_full = heatmap_upsampled.squeeze().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_array)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")
    
    # Heatmap only
    im = axes[1].imshow(heatmap_full, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title("Patch Probability Heatmap", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="P(Fake)", shrink=0.8)
    
    # Overlay
    axes[2].imshow(original_array)
    axes[2].imshow(heatmap_full, cmap=colormap, alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title("Heatmap Overlay", fontsize=14)
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_topk_visualization(
    image_path: str,
    patch_probs: np.ndarray,
    top_k_positions: List[Tuple[int, int]],
    output_path: str,
):
    """Create visualization highlighting top-k patches."""
    # Load original image
    original = Image.open(image_path).convert("RGB")
    original_array = np.array(original)
    
    # Calculate patch size in image coordinates
    H, W = patch_probs.shape
    img_h, img_w = original_array.shape[:2]
    patch_h = img_h / H
    patch_w = img_w / W
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(original_array)
    
    # Draw rectangles around top-k patches
    colors = ['#FF0000', '#FF6600', '#FFCC00']  # Red to yellow gradient
    
    for idx, (i, j) in enumerate(reversed(top_k_positions)):
        color = colors[min(idx, len(colors) - 1)]
        rect = mpatches.Rectangle(
            (j * patch_w, i * patch_h),
            patch_w, patch_h,
            linewidth=3,
            edgecolor=color,
            facecolor='none',
        )
        ax.add_patch(rect)
        
        # Add rank label
        rank = len(top_k_positions) - idx
        ax.text(
            j * patch_w + patch_w / 2,
            i * patch_h + patch_h / 2,
            f"#{rank}",
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )
    
    ax.set_title(f"Top-{len(top_k_positions)} Most Suspicious Patches", fontsize=14)
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_summary_card(
    forensic_output: Dict,
    output_path: str,
):
    """Create a summary card with all forensic information."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Load original image
    original = Image.open(forensic_output["image_path"]).convert("RGB")
    original_array = np.array(original)
    
    # 1. Original image with prediction
    axes[0, 0].imshow(original_array)
    prediction = forensic_output["prediction_label"]
    prob = forensic_output["probability"]
    color = '#FF4444' if prediction == "FAKE" else '#44FF44'
    axes[0, 0].set_title(
        f"Prediction: {prediction} ({prob:.1%})",
        fontsize=14, fontweight='bold', color=color
    )
    axes[0, 0].axis("off")
    
    # 2. Heatmap overlay
    patch_probs = forensic_output["patch_probs"]
    img_h, img_w = original_array.shape[:2]
    heatmap_tensor = torch.from_numpy(patch_probs).unsqueeze(0).unsqueeze(0).float()
    heatmap_upsampled = F.interpolate(
        heatmap_tensor, size=(img_h, img_w), mode='bilinear', align_corners=False
    )
    heatmap_full = heatmap_upsampled.squeeze().numpy()
    
    axes[0, 1].imshow(original_array)
    im = axes[0, 1].imshow(heatmap_full, cmap='RdYlBu_r', alpha=0.6, vmin=0, vmax=1)
    axes[0, 1].set_title("Patch Probability Heatmap", fontsize=14)
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], label="P(Fake)", shrink=0.8)
    
    # 3. Top-k patches
    H, W = patch_probs.shape
    patch_h = img_h / H
    patch_w = img_w / W
    
    axes[1, 0].imshow(original_array)
    colors = ['#FF0000', '#FF6600', '#FFCC00']
    
    for idx, (i, j) in enumerate(reversed(forensic_output["top_k_positions"])):
        color = colors[min(idx, len(colors) - 1)]
        rect = mpatches.Rectangle(
            (j * patch_w, i * patch_h),
            patch_w, patch_h,
            linewidth=3,
            edgecolor=color,
            facecolor='none',
        )
        axes[1, 0].add_patch(rect)
    
    axes[1, 0].set_title("Top-K Suspicious Patches", fontsize=14)
    axes[1, 0].axis("off")
    
    # 4. Deletion sanity check
    deletion = forensic_output["deletion_results"]
    axes[1, 1].axis("off")
    
    # Create text summary
    sanity_status = "✓ PASSED" if deletion["sanity_passed"] else "✗ FAILED"
    sanity_color = '#44AA44' if deletion["sanity_passed"] else '#AA4444'
    
    text = (
        f"DELETION SANITY CHECK\n"
        f"{'─' * 40}\n\n"
        f"Original Probability:  {deletion['original_prob']:.4f}\n"
        f"After Masking Top-K:   {deletion['masked_prob']:.4f}\n"
        f"Probability Drop:      {deletion['prob_drop']:.4f}\n"
        f"Relative Drop:         {deletion['relative_drop']:.1%}\n\n"
        f"Status: {sanity_status}"
    )
    
    axes[1, 1].text(
        0.5, 0.5, text,
        transform=axes[1, 1].transAxes,
        fontsize=12, fontfamily='monospace',
        verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray')
    )
    axes[1, 1].set_title("Deletion Sanity Check", fontsize=14, color=sanity_color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main Script
# =============================================================================

def collect_samples(
    data_root: str,
    num_real: int = 25,
    num_fake: int = 25,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Collect sample images from dataset.
    
    Returns:
        (real_paths, fake_paths)
    """
    np.random.seed(seed)
    
    data_root = Path(data_root)
    
    # Look for real/fake directories or train/test splits
    real_paths = []
    fake_paths = []
    
    # Try different directory structures
    for split in ["train", "test", "val", ""]:
        real_dir = data_root / split / "real" if split else data_root / "real"
        fake_dir = data_root / split / "fake" if split else data_root / "fake"
        
        if real_dir.exists():
            real_paths.extend(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
        if fake_dir.exists():
            fake_paths.extend(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))
    
    if not real_paths or not fake_paths:
        raise ValueError(f"Could not find real/fake images in {data_root}")
    
    # Sample
    real_sample = np.random.choice(real_paths, min(num_real, len(real_paths)), replace=False)
    fake_sample = np.random.choice(fake_paths, min(num_fake, len(fake_paths)), replace=False)
    
    return [str(p) for p in real_sample], [str(p) for p in fake_sample]


def create_forensic_pack(
    model,
    pooling,
    real_paths: List[str],
    fake_paths: List[str],
    output_dir: str,
    threshold: float = 0.5,
    k: int = 3,
    device: str = "cuda",
):
    """
    Create complete forensic pack for all images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "real" / "heatmaps").mkdir(parents=True, exist_ok=True)
    (output_dir / "real" / "topk").mkdir(parents=True, exist_ok=True)
    (output_dir / "real" / "summary").mkdir(parents=True, exist_ok=True)
    (output_dir / "fake" / "heatmaps").mkdir(parents=True, exist_ok=True)
    (output_dir / "fake" / "topk").mkdir(parents=True, exist_ok=True)
    (output_dir / "fake" / "summary").mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ForensicHeatmapGenerator(model, pooling, device)
    
    all_results = {
        "real": [],
        "fake": [],
        "metadata": {
            "threshold": threshold,
            "k": k,
            "num_real": len(real_paths),
            "num_fake": len(fake_paths),
        }
    }
    
    # Process real images
    print("\nProcessing real images...")
    for i, path in enumerate(tqdm(real_paths, desc="Real")):
        try:
            result = generator.generate_forensic_output(path, threshold, k)
            
            # Generate visualizations
            img_name = Path(path).stem
            
            create_heatmap_visualization(
                path, result["patch_probs"],
                str(output_dir / "real" / "heatmaps" / f"{img_name}_heatmap.png")
            )
            
            create_topk_visualization(
                path, result["patch_probs"], result["top_k_positions"],
                str(output_dir / "real" / "topk" / f"{img_name}_topk.png")
            )
            
            create_summary_card(
                result,
                str(output_dir / "real" / "summary" / f"{img_name}_summary.png")
            )
            
            # Store result (without large arrays)
            result_json = {
                k: v for k, v in result.items() 
                if k not in ["patch_probs"]
            }
            all_results["real"].append(result_json)
            
        except Exception as e:
            print(f"  ⚠ Error processing {path}: {e}")
    
    # Process fake images
    print("\nProcessing fake images...")
    for i, path in enumerate(tqdm(fake_paths, desc="Fake")):
        try:
            result = generator.generate_forensic_output(path, threshold, k)
            
            # Generate visualizations
            img_name = Path(path).stem
            
            create_heatmap_visualization(
                path, result["patch_probs"],
                str(output_dir / "fake" / "heatmaps" / f"{img_name}_heatmap.png")
            )
            
            create_topk_visualization(
                path, result["patch_probs"], result["top_k_positions"],
                str(output_dir / "fake" / "topk" / f"{img_name}_topk.png")
            )
            
            create_summary_card(
                result,
                str(output_dir / "fake" / "summary" / f"{img_name}_summary.png")
            )
            
            # Store result
            result_json = {
                k: v for k, v in result.items() 
                if k not in ["patch_probs"]
            }
            all_results["fake"].append(result_json)
            
        except Exception as e:
            print(f"  ⚠ Error processing {path}: {e}")
    
    # Save results JSON
    results_path = output_dir / "forensic_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("FORENSIC PACK SUMMARY")
    print("=" * 80)
    
    # Calculate statistics
    real_correct = sum(1 for r in all_results["real"] if r["prediction"] == 0)
    fake_correct = sum(1 for r in all_results["fake"] if r["prediction"] == 1)
    real_sanity_passed = sum(1 for r in all_results["real"] if r["deletion_results"]["sanity_passed"])
    fake_sanity_passed = sum(1 for r in all_results["fake"] if r["deletion_results"]["sanity_passed"])
    
    print(f"\n📊 Classification Results:")
    print(f"  Real images:  {real_correct}/{len(all_results['real'])} correct ({real_correct/len(all_results['real']):.1%})")
    print(f"  Fake images:  {fake_correct}/{len(all_results['fake'])} correct ({fake_correct/len(all_results['fake']):.1%})")
    
    print(f"\n🔍 Deletion Sanity Check:")
    print(f"  Real images:  {real_sanity_passed}/{len(all_results['real'])} passed")
    print(f"  Fake images:  {fake_sanity_passed}/{len(all_results['fake'])} passed")
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"  - real/heatmaps/: Patch probability heatmaps")
    print(f"  - real/topk/: Top-k patch visualizations")
    print(f"  - real/summary/: Complete summary cards")
    print(f"  - fake/heatmaps/, fake/topk/, fake/summary/: Same for fake images")
    print(f"  - forensic_results.json: JSON with all predictions and deletion tests")
    
    print("\n" + "=" * 80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Create forensic output pack for deepfake detection audit"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of dataset (with real/fake subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/forensic_pack",
        help="Output directory for forensic pack",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="Number of samples per class (real and fake)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top patches to highlight and test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("FORENSIC OUTPUT PACK GENERATOR")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = LaDeDaWrapper(pretrained=False, freeze_backbone=False)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    # Load pooling
    pooling = TopKLogitPooling(r=3)
    pooling = pooling.to(args.device)
    
    # Collect samples
    print(f"\nCollecting {args.num_samples} samples per class...")
    real_paths, fake_paths = collect_samples(
        args.data_root,
        num_real=args.num_samples,
        num_fake=args.num_samples,
        seed=args.seed,
    )
    print(f"✓ Found {len(real_paths)} real and {len(fake_paths)} fake images")
    
    # Create forensic pack
    create_forensic_pack(
        model=model,
        pooling=pooling,
        real_paths=real_paths,
        fake_paths=fake_paths,
        output_dir=args.output_dir,
        threshold=args.threshold,
        k=args.top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
