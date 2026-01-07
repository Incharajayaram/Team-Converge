#!/usr/bin/env python3
"""
Diagnostic Script: Teacher Model Quality Validation

CRITICAL: Before training student via distillation, verify teacher is good enough.
A weak teacher (AUC < 0.80) will cause student distillation to fail.

This script:
1. Loads teacher model
2. Evaluates on validation set
3. Computes per-layer attribution (which layers are confident?)
4. Identifies potential issues

Usage:
    python diagnostic_teacher_quality.py \
        --teacher-path weights/teacher/WildRF_LaDeDa.pth \
        --dataset-path data/test \
        --output results/teacher_quality.json
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerActivationHook:
    """Hook to capture activations from specific layers."""

    def __init__(self):
        self.activations = {}

    def register_hooks(self, model, layer_names):
        """Register hooks on specified layers."""
        for name, module in model.named_modules():
            if name in layer_names:
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook


def evaluate_teacher(model, dataloader, device):
    """
    Evaluate teacher model on validation set.

    Args:
        model: Teacher model
        dataloader: DataLoader with (images, labels)
        device: torch device

    Returns:
        dict with metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Forward pass
            if hasattr(model, 'model'):  # Wrapped model
                outputs = model.model(images)
            else:
                outputs = model(images)

            # Handle different output shapes
            if outputs.dim() == 4:  # (B, 1, H, W) - patch-level
                # Pool spatially to get image-level logit
                image_logits = outputs.view(outputs.size(0), -1).mean(dim=1)
            elif outputs.dim() == 2:  # (B, 1) - image-level
                image_logits = outputs.squeeze(1)
            else:
                image_logits = outputs

            # Convert to probabilities
            probs = torch.sigmoid(image_logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(image_logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    # Compute metrics
    try:
        auc = roc_auc_score(all_labels, all_logits)
    except:
        auc = 0.5

    accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Analyze logit distributions
    real_logits = all_logits[all_labels == 0]
    fake_logits = all_logits[all_labels == 1]

    metrics = {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1]),
        },
        'logits': {
            'real': {
                'mean': float(np.mean(real_logits)),
                'std': float(np.std(real_logits)),
                'min': float(np.min(real_logits)),
                'max': float(np.max(real_logits)),
            },
            'fake': {
                'mean': float(np.mean(fake_logits)),
                'std': float(np.std(fake_logits)),
                'min': float(np.min(fake_logits)),
                'max': float(np.max(fake_logits)),
            },
            'logit_range': float(np.max(all_logits) - np.min(all_logits)),
        },
    }

    return metrics, all_logits


def diagnose_teacher_issues(metrics):
    """
    Diagnose potential issues with teacher model.

    Args:
        metrics: dict from evaluate_teacher

    Returns:
        list of issues found
    """
    issues = []

    # Check AUC
    auc = metrics['auc']
    if auc < 0.80:
        issues.append(f"⚠️ CRITICAL: Teacher AUC too low ({auc:.3f}). "
                      f"Student distillation WILL FAIL. "
                      f"Try different teacher weights or dataset.")
    elif auc < 0.90:
        issues.append(f"⚠️ WARNING: Teacher AUC marginally acceptable ({auc:.3f}). "
                      f"Student AUC will be lower.")

    # Check logit distribution
    real_logits = metrics['logits']['real']
    fake_logits = metrics['logits']['fake']

    logit_separation = abs(real_logits['mean'] - fake_logits['mean'])
    if logit_separation < 0.5:
        issues.append(f"⚠️ WARNING: Poor logit separation ({logit_separation:.3f}). "
                      f"Classes not well-separated in teacher.")

    # Check scale mismatch
    logit_range = metrics['logits']['logit_range']
    if logit_range > 100:
        issues.append(f"⚠️ WARNING: Extreme logit range ({logit_range:.1f}). "
                      f"May cause scale mismatch during distillation. "
                      f"Use Adaptive Layer Normalization in distillation loss.")

    # Check class imbalance
    cm = metrics['confusion_matrix']
    real_acc = cm['true_negatives'] / (cm['true_negatives'] + cm['false_positives'])
    fake_acc = cm['true_positives'] / (cm['true_positives'] + cm['false_negatives'])

    if abs(real_acc - fake_acc) > 0.15:
        issues.append(f"⚠️ WARNING: Class imbalance (Real acc={real_acc:.3f}, "
                      f"Fake acc={fake_acc:.3f}). Teacher biased toward one class.")

    return issues


def main():
    parser = argparse.ArgumentParser(description='Validate teacher model quality')
    parser.add_argument('--teacher-path', required=True, help='Path to teacher model')
    parser.add_argument('--dataset-path', required=True, help='Path to validation dataset')
    parser.add_argument('--output', default='results/teacher_quality.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=16)

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load teacher (placeholder - adapt to your model)
    logger.info(f"Loading teacher from {args.teacher_path}")
    try:
        from models.reference.LaDeDa import LaDeDa9
        teacher = LaDeDa9(preprocess_type='NPR', num_classes=1, pool=True)
        state_dict = torch.load(args.teacher_path, map_location=device)
        teacher.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load teacher: {e}")
        return

    teacher = teacher.to(device)

    # Load validation data (placeholder - adapt to your data loading)
    logger.info(f"Loading validation dataset from {args.dataset_path}")
    # dataloader = load_validation_dataset(args.dataset_path, batch_size=args.batch_size)

    # Evaluate teacher
    logger.info("Evaluating teacher...")
    # metrics, logits = evaluate_teacher(teacher, dataloader, device)

    # Diagnose issues
    logger.info("\n" + "=" * 80)
    logger.info("TEACHER MODEL QUALITY ASSESSMENT")
    logger.info("=" * 80)

    # logger.info(f"AUC: {metrics['auc']:.3f}")
    # logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    # logger.info(f"Precision: {metrics['precision']:.3f}")
    # logger.info(f"Recall: {metrics['recall']:.3f}")

    # logger.info("\nLogit Statistics:")
    # logger.info(f"Real logits:  μ={metrics['logits']['real']['mean']:>8.2f}, "
    #            f"σ={metrics['logits']['real']['std']:>8.2f}, "
    #            f"range=[{metrics['logits']['real']['min']:>8.2f}, "
    #            f"{metrics['logits']['real']['max']:>8.2f}]")
    # logger.info(f"Fake logits:  μ={metrics['logits']['fake']['mean']:>8.2f}, "
    #            f"σ={metrics['logits']['fake']['std']:>8.2f}, "
    #            f"range=[{metrics['logits']['fake']['min']:>8.2f}, "
    #            f"{metrics['logits']['fake']['max']:>8.2f}]")

    # issues = diagnose_teacher_issues(metrics)
    # if issues:
    #    logger.info("\n⚠️ ISSUES FOUND:")
    #    for issue in issues:
    #        logger.info(f"  {issue}")
    # else:
    #    logger.info("\n✓ No issues found. Teacher is suitable for distillation.")

    # Save results
    # output_path = Path(args.output)
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(output_path, 'w') as f:
    #    json.dump({'metrics': metrics, 'issues': issues}, f, indent=2)

    # logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
