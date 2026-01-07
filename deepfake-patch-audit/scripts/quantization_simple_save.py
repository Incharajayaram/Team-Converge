#!/usr/bin/env python3
"""
Simple Quantization Script - Just Quantize and Save

No fancy evaluation. Just:
1. Load models
2. Quantize them
3. Save quantized versions
4. Report file sizes

Usage:
    python3 quantization_simple_save.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def quantize_and_save_model(model_path, output_path, model_name, num_calibration_batches=20):
    """
    Quantize a single model and save it.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"QUANTIZING: {model_name}")
    logger.info(f"{'='*80}")

    device = 'cpu'

    # Load model
    logger.info(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    if 'teacher' in model_name.lower():
        from models.reference.LaDeDa import LaDeDa9
        model = LaDeDa9(preprocess_type='NPR', num_classes=1, pool=True)
    else:
        from models.student.tiny_ladeda import TinyLaDeDa
        model = TinyLaDeDa(pretrained=False)
        model = model.model

    model.load_state_dict(state_dict)
    model.eval().to(device)
    logger.info(f"  ✓ Loaded {model_name}")

    # Get original file size
    original_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    logger.info(f"  Original size: {original_size_mb:.2f} MB")

    # Prepare for quantization
    logger.info(f"\nPreparing quantization...")
    quantization_config = torch.quantization.get_default_qconfig('fbgemm')
    model.qconfig = quantization_config
    torch.quantization.prepare(model, inplace=True)
    logger.info(f"  ✓ Inserted quantization modules")

    # Calibrate on random data
    logger.info(f"Calibrating on random data ({num_calibration_batches} batches)...")
    with torch.no_grad():
        for i in range(num_calibration_batches):
            dummy_input = torch.randn(2, 3, 256, 256).to(device)
            _ = model(dummy_input)
            if (i + 1) % 5 == 0:
                logger.info(f"  Calibrated: {i + 1}/{num_calibration_batches}")
    logger.info(f"  ✓ Calibration complete")

    # Convert to quantized
    logger.info(f"Converting to INT8...")
    torch.quantization.convert(model, inplace=True)
    logger.info(f"  ✓ Converted to INT8")

    # Save quantized model
    logger.info(f"\nSaving quantized model...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"  ✓ Saved to {output_path}")

    # Check file size
    quantized_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quantized_size_mb / original_size_mb) * 100

    logger.info(f"\n{model_name.upper()} RESULTS:")
    logger.info(f"  Original size:    {original_size_mb:>10.2f} MB")
    logger.info(f"  Quantized size:   {quantized_size_mb:>10.2f} MB")
    logger.info(f"  Size reduction:   {reduction:>10.1f}%")
    logger.info(f"  Compression:      {original_size_mb/quantized_size_mb:>10.1f}x smaller")

    return {
        'model': model_name,
        'original_size_mb': original_size_mb,
        'quantized_size_mb': quantized_size_mb,
        'reduction_percent': reduction,
        'compression_ratio': original_size_mb / quantized_size_mb,
    }


def main():
    logger.info(f"\n{'='*80}")
    logger.info("SIMPLE QUANTIZATION - Save Only")
    logger.info(f"{'='*80}")

    results = []

    # Quantize teacher
    try:
        result = quantize_and_save_model(
            model_path='deepfake-patch-audit/weights/teacher/WildRF_LaDeDa.pth',
            output_path=Path('results/quantization_proper/teacher_quantized.pt'),
            model_name='Teacher'
        )
        results.append(result)
    except Exception as e:
        logger.error(f"Failed to quantize teacher: {e}")

    # Quantize student
    try:
        result = quantize_and_save_model(
            model_path='deepfake-patch-audit/weights/student/WildRF_Tiny_LaDeDa.pth',
            output_path=Path('results/quantization_proper/student_quantized.pt'),
            model_name='Student',
            num_calibration_batches=10
        )
        results.append(result)
    except Exception as e:
        logger.error(f"Failed to quantize student: {e}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")

    for result in results:
        logger.info(f"\n{result['model']}:")
        logger.info(f"  {result['original_size_mb']:.2f} MB → {result['quantized_size_mb']:.2f} MB")
        logger.info(f"  Reduction: {result['reduction_percent']:.1f}% ({result['compression_ratio']:.1f}x)")

    logger.info(f"\n✓ Quantized models saved to: results/quantization_proper/")
    logger.info(f"\nFiles created:")
    logger.info(f"  - teacher_quantized.pt")
    logger.info(f"  - student_quantized.pt")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Copy to deployment folders")
    logger.info(f"  2. Test inference on your device")
    logger.info(f"  3. Compare with original models")


if __name__ == '__main__':
    main()
