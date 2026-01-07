#!/usr/bin/env python3
"""
Diagnostic Script: Aggregation Strategy Comparison

Compares different patch aggregation methods to identify which is most robust.
Tests: Fixed top-k, dynamic threshold, percentile-based, attention-weighted.

Usage:
    python diagnostic_aggregation_strategies.py \
        --model-path weights/student/ForenSynth_Tiny_LaDeDa.pth \
        --dataset-path data/test \
        --output results/aggregation_comparison.json
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedTopKPooling(nn.Module):
    """Fixed top-k pooling (current implementation)."""

    def __init__(self, k_ratio=0.1, min_k=5):
        super().__init__()
        self.k_ratio = k_ratio
        self.min_k = min_k

    def forward(self, patch_logits):
        """
        Args:
            patch_logits: (B, 1, H, W) or (B, H*W)

        Returns:
            image_logits: (B, 1)
        """
        if patch_logits.dim() == 4:
            # (B, C, H, W) -> (B, C*H*W)
            batch_size = patch_logits.size(0)
            patch_logits_flat = patch_logits.view(batch_size, -1)
        else:
            patch_logits_flat = patch_logits

        n_patches = patch_logits_flat.size(1)
        k = max(self.min_k, int(np.ceil(self.k_ratio * n_patches)))

        # Select top-k
        topk_values, _ = torch.topk(patch_logits_flat, k, dim=1)
        image_logits = topk_values.mean(dim=1, keepdim=True)

        return image_logits


class DynamicThresholdPooling(nn.Module):
    """Dynamic threshold-based pooling (proposed improvement)."""

    def __init__(self, percentile=90):
        super().__init__()
        self.percentile = percentile

    def forward(self, patch_logits):
        """
        Args:
            patch_logits: (B, 1, H, W) or (B, H*W)

        Returns:
            image_logits: (B, 1)
        """
        if patch_logits.dim() == 4:
            batch_size = patch_logits.size(0)
            patch_logits_flat = patch_logits.view(batch_size, -1)
        else:
            batch_size = patch_logits.size(0)
            patch_logits_flat = patch_logits

        # Dynamic threshold per image
        thresholds = torch.quantile(patch_logits_flat, self.percentile / 100.0, dim=1, keepdim=True)

        # Select patches above threshold
        mask = patch_logits_flat > thresholds
        selected = torch.where(mask, patch_logits_flat, torch.zeros_like(patch_logits_flat))

        # Mean of selected patches (or zero if none selected)
        image_logits = torch.sum(selected, dim=1, keepdim=True) / torch.clamp(
            torch.sum(mask, dim=1, keepdim=True), min=1
        )

        return image_logits


class PercentilePooling(nn.Module):
    """Percentile-based pooling (simpler threshold variant)."""

    def __init__(self, percentile_k=10):
        super().__init__()
        self.percentile_k = percentile_k  # e.g., 10 = top 10%

    def forward(self, patch_logits):
        if patch_logits.dim() == 4:
            batch_size = patch_logits.size(0)
            patch_logits_flat = patch_logits.view(batch_size, -1)
        else:
            batch_size = patch_logits.size(0)
            patch_logits_flat = patch_logits

        # Percentile threshold
        percentile = 100 - self.percentile_k
        thresholds = torch.quantile(patch_logits_flat, percentile / 100.0, dim=1, keepdim=True)

        # Select patches above threshold
        topk_values = patch_logits_flat[patch_logits_flat > thresholds]
        image_logits = torch.mean(topk_values.view(batch_size, -1), dim=1, keepdim=True)

        return image_logits


class MeanPooling(nn.Module):
    """Simple global mean pooling (baseline)."""

    def forward(self, patch_logits):
        if patch_logits.dim() == 4:
            return patch_logits.view(patch_logits.size(0), -1).mean(dim=1, keepdim=True)
        else:
            return patch_logits.mean(dim=1, keepdim=True)


class MedianPooling(nn.Module):
    """Median pooling (robust to outliers)."""

    def forward(self, patch_logits):
        if patch_logits.dim() == 4:
            patch_logits_flat = patch_logits.view(patch_logits.size(0), -1)
        else:
            patch_logits_flat = patch_logits

        image_logits = torch.median(patch_logits_flat, dim=1, keepdim=True).values
        return image_logits


def compare_aggregation_methods(patch_logits_list, labels_list):
    """
    Compare different aggregation methods on patch logits.

    Args:
        patch_logits_list: List of (B, 1, H, W) tensors
        labels_list: List of binary labels

    Returns:
        dict with AUC scores for each method
    """
    methods = {
        'fixed_10%': FixedTopKPooling(k_ratio=0.1, min_k=5),
        'fixed_5%': FixedTopKPooling(k_ratio=0.05, min_k=3),
        'dynamic_90': DynamicThresholdPooling(percentile=90),
        'dynamic_75': DynamicThresholdPooling(percentile=75),
        'percentile_10': PercentilePooling(percentile_k=10),
        'percentile_5': PercentilePooling(percentile_k=5),
        'mean': MeanPooling(),
        'median': MedianPooling(),
    }

    results = {}

    for method_name, pooler in methods.items():
        logger.info(f"Testing {method_name}...")

        all_preds = []
        all_labels = []

        for patch_logits, labels in zip(patch_logits_list, labels_list):
            # Apply pooling
            image_logits = pooler(patch_logits)

            # Convert to probabilities
            probs = torch.sigmoid(image_logits).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()

            all_preds.extend(preds)
            all_labels.extend(labels.flatten())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5

        accuracy = accuracy_score(all_labels, all_preds)

        results[method_name] = {
            'auc': float(auc),
            'accuracy': float(accuracy),
        }

        logger.info(f"  AUC: {auc:.3f}, Accuracy: {accuracy:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare aggregation strategies')
    parser.add_argument('--output', default='results/aggregation_comparison.json')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for testing')

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("AGGREGATION STRATEGY COMPARISON")
    logger.info("=" * 80)

    if args.synthetic:
        logger.info("Using synthetic data for testing...")

        # Create synthetic patch logits
        batch_size = 32
        height, width = 126, 126
        n_batches = 10

        patch_logits_list = []
        labels_list = []

        for i in range(n_batches):
            # Real patches: mostly negative logits
            if i % 2 == 0:
                logits = torch.randn(batch_size, 1, height, width) * 0.5 - 1.0
                label = 0  # Real
            else:
                # Fake patches: mostly positive logits
                logits = torch.randn(batch_size, 1, height, width) * 0.8 + 0.5
                label = 1  # Fake

            patch_logits_list.append(logits)
            labels_list.append(np.array([label] * batch_size))

        # Compare methods
        results = compare_aggregation_methods(patch_logits_list, labels_list)

        # Save results
        logger.info(f"\nSaving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Method':<20} {'AUC':<10} {'Accuracy':<10}")
        logger.info("-" * 40)

        sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
        for method, metrics in sorted_results:
            logger.info(f"{method:<20} {metrics['auc']:<10.3f} {metrics['accuracy']:<10.3f}")

        # Recommendations
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 80)

        best_method = sorted_results[0][0]
        best_auc = sorted_results[0][1]['auc']

        logger.info(f"✓ Best method: {best_method} (AUC={best_auc:.3f})")
        logger.info(f"  Current method (fixed_10%): AUC={results['fixed_10%']['auc']:.3f}")

        improvement = best_auc - results['fixed_10%']['auc']
        if improvement > 0.01:
            logger.info(f"  ⚠️ Improvement potential: +{improvement:.1%}")
        else:
            logger.info(f"  ✓ Current method is near-optimal")


if __name__ == '__main__':
    main()
