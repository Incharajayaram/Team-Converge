#!/usr/bin/env python3
"""
Complete deployment pipeline: PyTorch → ONNX → TFLite → Quantized TFLite

Pipeline stages:
1. Load trained PyTorch student model
2. Export to ONNX format
3. Convert ONNX to TFLite
4. Apply post-training quantization (dynamic range)
5. Verify all models and compare file sizes
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student.tiny_ladeda import TinyLaDeDa
from export.onnx_exporter import ONNXExporter
from export.tflite_converter import TFLiteConverter
from quantization.improved_dynamic_range import ImprovedDynamicRangeQuantizer


def load_trained_model(model_path: str, device: str = "cuda"):
    """Load trained student model."""
    print("\n" + "=" * 80)
    print("LOADING TRAINED MODEL")
    print("=" * 80)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return None

    try:
        student = TinyLaDeDa(pretrained=False)
        state_dict = torch.load(model_path, map_location="cpu")
        student.load_state_dict(state_dict)
        student = student.to(device)
        student.eval()

        print(f"✓ Model loaded from {model_path}")
        print(f"  Parameters: {student.count_parameters()}")

        return student

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def apply_quantization(model: torch.nn.Module, strategy: str = "dynamic") -> dict:
    """Apply quantization to model weights."""
    print("\n" + "=" * 80)
    print("APPLYING QUANTIZATION")
    print("=" * 80)

    if strategy == "dynamic":
        quantizer = ImprovedDynamicRangeQuantizer(
            bits=8,
            symmetric=False,
            per_channel=True,
            clip_outliers=True,
            clip_percentile=99.9,
        )

        print("Configuration:")
        print(f"  Bits: 8 (int8)")
        print(f"  Mode: Dynamic range (asymmetric)")
        print(f"  Per-channel: Yes")
        print(f"  Clip outliers: Yes (99.9th percentile)")

        quantized_model, quant_params = quantizer.quantize_model(model)

        # Print quantization report
        print(quantizer.get_quantization_report(quant_params))

        return {"model": quantized_model, "quantizer": quantizer, "params": quant_params}

    else:
        print(f"⚠ Unknown quantization strategy: {strategy}")
        return None


def create_deployment_pipeline():
    """Create the complete deployment pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete deployment pipeline: PyTorch → ONNX → TFLite → Quantized"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained PyTorch model (.pt file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployment/pi_server/models",
        help="Output directory for exported models (default: deployment/pi_server/models for Flask server)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda/cpu)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "dynamic"],
        help="Quantization strategy (default: none for ONNX on Raspberry Pi)",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (useful if already exported)",
    )
    parser.add_argument(
        "--skip-tflite",
        action="store_true",
        default=True,
        help="Skip TFLite conversion (default: True, only ONNX for Flask server)",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        default=True,
        help="Skip post-training quantization (default: True for simpler ONNX)",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Export only ONNX format (recommended for Flask server on Raspberry Pi)",
    )

    return parser


def main():
    """Run complete deployment pipeline."""
    parser = create_deployment_pipeline()
    args = parser.parse_args()

    # Handle --onnx-only flag
    if args.onnx_only:
        args.skip_tflite = True
        args.skip_quantization = True
        args.quantization = "none"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("DEEPFAKE DETECTION MODEL - DEPLOYMENT PIPELINE")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Quantization: {args.quantization}")
    print(f"Device: {args.device}")
    print(f"ONNX Only: {args.onnx_only}")

    # Step 1: Load trained model
    model = load_trained_model(args.model, args.device)
    if model is None:
        return False

    # Step 2: Apply quantization
    if args.quantization != "none":
        quant_result = apply_quantization(model, args.quantization)
        if quant_result is None:
            return False
        quantized_model = quant_result["model"]
    else:
        quantized_model = model

    # Step 3: Export to ONNX
    onnx_path = output_dir / "deepfake_detector.onnx"
    if not args.skip_onnx:
        exporter = ONNXExporter(quantized_model, args.device)
        success = exporter.export(
            str(onnx_path),
            input_shape=(1, 3, 256, 256),
            opset_version=12,
            verbose=True,
        )
        if not success:
            return False

        # Verify ONNX model
        exporter.verify_onnx_model(str(onnx_path))

        # Get ONNX info
        onnx_info = exporter.get_onnx_model_info(str(onnx_path))
        if onnx_info:
            print("\nONNX Model Info:")
            print(f"  Inputs: {onnx_info['inputs']}")
            print(f"  Outputs: {onnx_info['outputs']}")
            print(f"  Nodes: {onnx_info['num_nodes']}")

    # Step 4: Convert to TFLite
    tflite_path = output_dir / "deepfake_detector.tflite"
    if not args.skip_tflite:
        if not onnx_path.exists():
            print(f"✗ ONNX model not found: {onnx_path}")
            return False

        converter = TFLiteConverter(str(onnx_path), verbose=True)
        success = converter.convert_to_tflite(
            str(tflite_path),
            quantization_mode="dynamic" if args.quantization == "dynamic" else "none",
        )
        if not success:
            print("⚠ TFLite conversion failed (may require additional dependencies)")
            print("  To use TFLite, install: pip install onnx-tf tensorflow")
        else:
            # Verify TFLite model
            test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            converter.verify_tflite_model(str(tflite_path), test_input)

            # Get TFLite info
            tflite_info = converter.get_tflite_info(str(tflite_path))
            if tflite_info:
                print("\nTFLite Model Info:")
                print(f"  Inputs: {tflite_info['inputs']}")
                print(f"  Outputs: {tflite_info['outputs']}")
                print(f"  Size: {tflite_info['file_size_mb']:.2f} MB")

    # Step 5: Summary and file sizes
    print("\n" + "=" * 80)
    print("DEPLOYMENT SUMMARY")
    print("=" * 80)

    files_info = []

    # PyTorch model size
    pytorch_path = Path(args.model)
    if pytorch_path.exists():
        pytorch_size = pytorch_path.stat().st_size / (1024 * 1024)
        files_info.append(("PyTorch (.pt)", pytorch_size))

    # ONNX model size
    if onnx_path.exists():
        onnx_size = onnx_path.stat().st_size / (1024 * 1024)
        files_info.append(("ONNX (.onnx)", onnx_size))

    # TFLite model size
    if tflite_path.exists():
        tflite_size = tflite_path.stat().st_size / (1024 * 1024)
        files_info.append(("TFLite (.tflite)", tflite_size))

    print("\nModel Sizes:")
    for name, size in files_info:
        print(f"  {name}: {size:.2f} MB")

    # Compression statistics
    if len(files_info) > 1:
        original_size = files_info[0][1]
        print("\nCompression Ratios (vs PyTorch):")
        for i in range(1, len(files_info)):
            name, size = files_info[i]
            ratio = (1 - size / original_size) * 100
            print(f"  {name}: {ratio:.1f}% smaller")

    print("\n" + "=" * 80)
    print("✓ DEPLOYMENT PIPELINE COMPLETE")
    print("=" * 80)

    print("\nDeployed Models:")
    for name, path in [
        ("ONNX", onnx_path),
        ("TFLite", tflite_path),
    ]:
        if path.exists():
            print(f"  {name}: {path}")

    print("\nNext Steps:")
    print("  1. Test models with sample data")
    print("  2. Benchmark performance on target hardware")
    print("  3. Deploy to mobile/edge devices")
    print("  4. Monitor accuracy and latency in production")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
