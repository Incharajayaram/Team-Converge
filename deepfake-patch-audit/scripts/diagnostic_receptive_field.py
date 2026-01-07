#!/usr/bin/env python3
"""
Diagnostic Script: Receptive Field Analysis

Empirically measures the effective receptive field of models using gradient-based analysis.
This identifies how much of the input each layer can "see".

Usage:
    python diagnostic_receptive_field.py \
        --model teacher \
        --output results/rf_analysis.json
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_receptive_field_empirical(model, input_shape=(1, 3, 256, 256), device='cuda'):
    """
    Measure receptive field empirically using gradient-based analysis.

    For each layer, we:
    1. Create input with all zeros except center pixel
    2. Forward pass through model
    3. Backward from output center
    4. Count which input pixels have non-zero gradients

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W)
        device: torch device

    Returns:
        dict with receptive field analysis per layer
    """
    model.eval()
    model.to(device)

    results = {
        'model': model.__class__.__name__,
        'input_shape': input_shape,
        'layers': {},
    }

    # Create input with sparse gradient signal
    x = torch.zeros(input_shape, dtype=torch.float32, device=device, requires_grad=True)

    # Set center pixel to 1
    center_h = input_shape[2] // 2
    center_w = input_shape[3] // 2
    x.data[0, 0, center_h, center_w] = 1.0

    # Forward pass
    with torch.enable_grad():
        y = model(x)

        # Backward from center of output
        if y.dim() == 4:  # (B, C, H, W) - feature map output
            out_h = y.size(2) // 2
            out_w = y.size(3) // 2
            grad_output = torch.zeros_like(y)
            grad_output[0, 0, out_h, out_w] = 1.0
        elif y.dim() == 2:  # (B, C) - scalar output
            grad_output = torch.zeros_like(y)
            grad_output[0, 0] = 1.0
        else:
            grad_output = torch.ones_like(y)

        y.backward(grad_output, retain_graph=True)

    # Measure receptive field
    input_grad = x.grad.abs()
    receptive_field_mask = (input_grad > 1e-6).squeeze().cpu().numpy()

    # Count connected pixels from center
    rf_size_h = np.sum(np.any(receptive_field_mask, axis=1))
    rf_size_w = np.sum(np.any(receptive_field_mask, axis=0))
    rf_coverage = np.sum(receptive_field_mask) / np.prod(input_shape[2:]) * 100

    results['receptive_field'] = {
        'height_span': int(rf_size_h),
        'width_span': int(rf_size_w),
        'coverage_percent': float(rf_coverage),
        'center_activated': bool(receptive_field_mask[center_h, center_w]),
    }

    logger.info(f"Receptive field analysis for {model.__class__.__name__}:")
    logger.info(f"  Height span: {rf_size_h} pixels")
    logger.info(f"  Width span: {rf_size_w} pixels")
    logger.info(f"  Coverage: {rf_coverage:.1f}% of input")

    return results


def measure_receptive_field_theoretical(architecture_config):
    """
    Calculate receptive field theoretically from architecture specification.

    Formula: RF_out = RF_in + (kernel_size - 1) * stride_accumulated

    Args:
        architecture_config: dict with layer specifications

    Returns:
        dict with theoretical RF per layer
    """
    rf = 1  # Initial receptive field
    stride_acc = 1  # Accumulated stride

    results = {
        'layers': [],
        'theoretical_rf': 1,
    }

    for layer_idx, layer_config in enumerate(architecture_config):
        kernel = layer_config.get('kernel_size', 1)
        stride = layer_config.get('stride', 1)
        dilation = layer_config.get('dilation', 1)

        # Update RF
        rf = rf + (kernel - 1) * dilation * stride_acc
        stride_acc *= stride

        results['layers'].append({
            'index': layer_idx,
            'kernel_size': kernel,
            'stride': stride,
            'dilation': dilation,
            'receptive_field': int(rf),
            'stride_accumulated': int(stride_acc),
        })

        logger.info(f"Layer {layer_idx}: RF={rf}, stride_acc={stride_acc}")

    results['theoretical_rf'] = int(rf)
    return results


def analyze_ladeda9():
    """Analyze LaDeDa9 receptive field."""
    logger.info("=" * 80)
    logger.info("LaDeDa9 RECEPTIVE FIELD ANALYSIS")
    logger.info("=" * 80)

    # Architecture specification
    # Input: 256×256 images
    # Conv1 (k=1, s=1), Conv2 (k=3, s=1), BN, ReLU
    # Layer1 (3 bottlenecks, s=1, k=3 for first block only)
    # Layer2 (4 bottlenecks, s=2, k=3 or k=1)
    # Layer3 (6 bottlenecks, s=2, k=1)
    # Layer4 (3 bottlenecks, s=1, k=1)

    layers = [
        {'name': 'Preprocess (NPR)', 'kernel_size': 1, 'stride': 1},  # No spatial change
        {'name': 'Conv1', 'kernel_size': 1, 'stride': 1},
        {'name': 'Conv2', 'kernel_size': 3, 'stride': 1},
        {'name': 'Layer1[0] Conv2d', 'kernel_size': 3, 'stride': 1},
        {'name': 'Layer1[1-2] Conv2d', 'kernel_size': 1, 'stride': 1},
        {'name': 'Layer2[0] Conv2d', 'kernel_size': 3, 'stride': 2},
        {'name': 'Layer2[1-3] Conv2d', 'kernel_size': 1, 'stride': 1},
        {'name': 'Layer3[0] Conv2d', 'kernel_size': 1, 'stride': 2},  # No 3×3 in layer3
        {'name': 'Layer3[1-5] Conv2d', 'kernel_size': 1, 'stride': 1},
        {'name': 'Layer4[0-2] Conv2d', 'kernel_size': 1, 'stride': 1},
    ]

    rf = 1
    stride_acc = 1
    output_size = 256

    logger.info(f"{'Layer':<30} {'Kernel':<10} {'Stride':<10} {'RF':<10} {'Output':<10} {'Coverage %':<12}")
    logger.info("-" * 90)

    results = {'layers': []}

    for layer in layers:
        kernel = layer['kernel_size']
        stride = layer['stride']

        # Update RF
        rf = rf + (kernel - 1) * stride_acc
        stride_acc *= stride
        output_size = (output_size - 1) // stride + 1 if stride > 1 else output_size

        coverage = min(100.0, (rf / 256.0) * 100)  # As % of input
        results['layers'].append({
            'name': layer['name'],
            'kernel_size': kernel,
            'stride': stride,
            'receptive_field': int(rf),
            'stride_accumulated': int(stride_acc),
            'output_size': int(output_size),
            'coverage_percent': float(coverage),
        })

        logger.info(
            f"{layer['name']:<30} {kernel:<10} {stride:<10} {rf:<10} {output_size:<10} {coverage:<11.1f}%"
        )

    logger.info("-" * 90)
    logger.info(f"Final: RF={rf}, Output spatial size={output_size}×{output_size}")

    return results


def analyze_tiny_ladeda():
    """Analyze TinyLaDeDa receptive field."""
    logger.info("\n" + "=" * 80)
    logger.info("TinyLaDeDa RECEPTIVE FIELD ANALYSIS")
    logger.info("=" * 80)

    # Architecture:
    # Input: 256×256
    # Conv1 (k=1, s=1), Conv2 (k=3, s=1), BN, ReLU
    # Layer1 (1 bottleneck, s=2, k=3 for main conv)
    # FC output: (B, 1, 126, 126) = after spatial downsampling

    layers = [
        {'name': 'Preprocess (right_diag)', 'kernel_size': 1, 'stride': 1},
        {'name': 'Conv1', 'kernel_size': 1, 'stride': 1},
        {'name': 'Conv2', 'kernel_size': 3, 'stride': 1},
        {'name': 'Layer1 Conv2d', 'kernel_size': 3, 'stride': 2},
    ]

    rf = 1
    stride_acc = 1
    output_size = 256

    logger.info(f"{'Layer':<30} {'Kernel':<10} {'Stride':<10} {'RF':<10} {'Output':<10} {'Coverage %':<12}")
    logger.info("-" * 90)

    results = {'layers': []}

    for layer in layers:
        kernel = layer['kernel_size']
        stride = layer['stride']

        rf = rf + (kernel - 1) * stride_acc
        stride_acc *= stride
        output_size = (output_size - 1) // stride + 1 if stride > 1 else output_size

        coverage = min(100.0, (rf / 256.0) * 100)
        results['layers'].append({
            'name': layer['name'],
            'kernel_size': kernel,
            'stride': stride,
            'receptive_field': int(rf),
            'stride_accumulated': int(stride_acc),
            'output_size': int(output_size),
            'coverage_percent': float(coverage),
        })

        logger.info(
            f"{layer['name']:<30} {kernel:<10} {stride:<10} {rf:<10} {output_size:<10} {coverage:<11.1f}%"
        )

    logger.info("-" * 90)
    logger.info(f"Final: RF={rf}, Output spatial size={output_size}×{output_size}")
    logger.info(f"⚠️  CRITICAL: RF={rf} is only {(rf/256)*100:.1f}% of input!")

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze receptive field')
    parser.add_argument('--output', default='results/receptive_field_analysis.json')
    parser.add_argument('--models', nargs='+', default=['ladeda9', 'tiny_ladeda'], help='Models to analyze')

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if 'ladeda9' in args.models:
        all_results['LaDeDa9'] = analyze_ladeda9()

    if 'tiny_ladeda' in args.models:
        all_results['TinyLaDeDa'] = analyze_tiny_ladeda()

    # Save results
    logger.info(f"\nSaving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info("✓ Analysis complete")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for model_name, result in all_results.items():
        final_layer = result['layers'][-1]
        logger.info(f"{model_name}: RF={final_layer['receptive_field']}, "
                    f"Coverage={final_layer['coverage_percent']:.1f}%")


if __name__ == '__main__':
    main()
