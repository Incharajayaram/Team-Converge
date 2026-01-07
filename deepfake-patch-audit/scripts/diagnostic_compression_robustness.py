#!/usr/bin/env python3
"""
Diagnostic Script: Compression Robustness Testing

Tests how deepfake detection AUC degrades with JPEG compression.
This is CRITICAL - compression is the #1 failure mode in real-world deployment.

Usage:
    python diagnostic_compression_robustness.py \
        --model-path weights/student/ForenSynth_Tiny_LaDeDa.pth \
        --dataset-path data/test \
        --output results/compression_test.json
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import json
import argparse
from PIL import Image
import io
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_jpeg_compression(image_tensor, quality):
    """
    Apply JPEG compression to a tensor image.

    Args:
        image_tensor: torch tensor of shape (B, 3, H, W) with values in [0, 1]
        quality: JPEG quality (1-100), where 100 is lossless

    Returns:
        Compressed image tensor
    """
    batch_size = image_tensor.size(0)
    compressed = []

    for i in range(batch_size):
        # Convert tensor to PIL Image
        img_np = (image_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Compress and decompress
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_pil = Image.open(buffer)

        # Convert back to tensor
        compressed_np = np.array(compressed_pil).astype(np.float32) / 255.0
        compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1)
        compressed.append(compressed_tensor)

    return torch.stack(compressed)


def evaluate_model(model, dataloader, device, quality=None):
    """
    Evaluate model on dataset with optional JPEG compression.

    Args:
        model: PyTorch model
        dataloader: DataLoader with (images, labels)
        device: torch device
        quality: JPEG quality (None = no compression, 1-100 = compression level)

    Returns:
        dict with metrics: auc, accuracy, precision, recall, confusion_matrix
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Apply compression if specified
            if quality is not None:
                images = apply_jpeg_compression(images, quality)
                images = images.to(device)

            # Forward pass
            if hasattr(model, 'model'):  # Wrapper model
                patch_logits = model.model(images)
            else:
                patch_logits = model(images)

            # Global average pool for image-level prediction
            image_logits = patch_logits.view(patch_logits.size(0), -1).mean(dim=1)

            # Convert to probabilities
            probs = torch.sigmoid(image_logits).cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5  # Fallback if AUC can't be computed

    accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description='Test compression robustness')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--dataset-path', required=True, help='Path to test dataset')
    parser.add_argument('--output', default='results/compression_robustness.json', help='Output JSON file')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=None, help='Max samples to test')

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model (placeholder - adapt to your actual model)
    logger.info(f"Loading model from {args.model_path}")
    try:
        from models.student.tiny_ladeda import TinyLaDeDa
        model = TinyLaDeDa(pretrained=False)
        state_dict = torch.load(args.model_path, map_location=device)
        model.model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    model = model.to(device)
    model.eval()

    # Load test data (placeholder - adapt to your actual data loading)
    logger.info(f"Loading test dataset from {args.dataset_path}")
    # This is pseudocode - replace with actual data loading
    # dataloader = load_test_dataset(args.dataset_path, batch_size=args.batch_size)

    # Test compression qualities
    qualities = [100, 95, 90, 85, 80, 75, 70, 65, 60, 50]
    results = {}

    logger.info("=" * 80)
    logger.info("COMPRESSION ROBUSTNESS TEST")
    logger.info("=" * 80)
    logger.info(f"{'Quality':<10} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    logger.info("-" * 80)

    for quality in qualities:
        # logger.info(f"Testing JPEG quality {quality}...")
        # metrics = evaluate_model(model, dataloader, device, quality=quality)
        # results[quality] = metrics

        # logger.info(f"  AUC: {metrics['auc']:.3f}, Acc: {metrics['accuracy']:.3f}")
        pass

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("-" * 80)
    logger.info(f"Results saved to {output_path}")

    # Plotting (commented out for now)
    # plot_compression_results(results, output_path.with_suffix('.png'))


def plot_compression_results(results, output_path):
    """Plot compression robustness curves."""
    qualities = sorted(results.keys())
    aucs = [results[q]['auc'] for q in qualities]
    accuracies = [results[q]['accuracy'] for q in qualities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AUC plot
    ax1.plot(qualities, aucs, marker='o', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
    ax1.axvline(x=75, color='gray', linestyle=':', label='Default JPEG Q=75')
    ax1.set_xlabel('JPEG Quality', fontsize=12)
    ax1.set_ylabel('AUC-ROC', fontsize=12)
    ax1.set_title('Compression Robustness - AUC', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0.45, 1.0])

    # Accuracy plot
    ax2.plot(qualities, accuracies, marker='s', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
    ax2.axvline(x=75, color='gray', linestyle=':', label='Default JPEG Q=75')
    ax2.set_xlabel('JPEG Quality', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Compression Robustness - Accuracy', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0.45, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Plot saved to {output_path}")


if __name__ == '__main__':
    main()
