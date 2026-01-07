"""TFLite conversion utilities for Phase 5 experiments.

Converts PyTorch models to TFLite format with optional quantization.

Pipeline: PyTorch -> ONNX -> TFLite

Usage:
    from ecdd_core.export.tflite_converter import convert_pytorch_to_tflite
    
    result = convert_pytorch_to_tflite(
        pytorch_model_path=Path("model.pth"),
        output_path=Path("model.tflite"),
        config=TFLiteConversionConfig(),
        model_class=LaDeDaResNet50,
    )
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np


@dataclass
class TFLiteConversionConfig:
    """Configuration for TFLite conversion."""
    input_shape: tuple = (1, 3, 256, 256)  # NCHW (PyTorch format)
    quantization: Literal["none", "dynamic", "int8"] = "none"
    representative_dataset_path: Optional[Path] = None  # For int8 quantization
    optimization_target: Literal["default", "low_latency", "low_memory"] = "default"
    opset_version: int = 12  # ONNX opset version


def export_pytorch_to_onnx(
    model,
    onnx_path: Path,
    input_shape: tuple = (1, 3, 256, 256),
    opset_version: int = 12,
) -> dict:
    """Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model (must be in eval mode)
        onnx_path: Path to save ONNX model
        input_shape: Input tensor shape (NCHW)
        opset_version: ONNX opset version
    
    Returns:
        Dictionary with export metadata
    """
    import torch
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Get output names from model signature
    with torch.no_grad():
        outputs = model(dummy_input)
    
    if isinstance(outputs, tuple):
        output_names = [f"output_{i}" for i in range(len(outputs))]
    else:
        output_names = ["output"]
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    return {
        "onnx_path": str(onnx_path),
        "size_mb": onnx_size_mb,
        "input_shape": input_shape,
        "output_names": output_names,
        "opset_version": opset_version,
    }


def convert_onnx_to_tflite(
    onnx_model_path: Path,
    output_path: Path,
    config: TFLiteConversionConfig,
    representative_data_gen: Optional[Callable] = None,
) -> dict:
    """Convert ONNX model to TFLite.
    
    Args:
        onnx_model_path: Path to ONNX model
        output_path: Path to save TFLite model
        config: Conversion configuration
        representative_data_gen: Generator function for int8 quantization
    
    Returns:
        Dictionary with conversion metadata
    """
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
    except ImportError as e:
        raise ImportError(
            f"TFLite conversion requires TensorFlow and onnx-tf. "
            f"Install: pip install tensorflow onnx onnx-tf\n"
            f"Missing: {e}"
        )
    
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_model_path))
    
    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Save as SavedModel
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = Path(tmpdir) / "saved_model"
        tf_rep.export_graph(str(saved_model_path))
        
        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        
        # Apply quantization settings
        if config.quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif config.quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_data_gen is not None:
                converter.representative_dataset = representative_data_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.float32
        
        # Apply optimization target
        if config.optimization_target == "low_latency":
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        elif config.optimization_target == "low_memory":
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    
    tflite_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    return {
        "tflite_path": str(output_path),
        "size_mb": tflite_size_mb,
        "quantization": config.quantization,
        "optimization_target": config.optimization_target,
    }


def convert_pytorch_to_tflite(
    pytorch_model_path: Path,
    output_path: Path,
    config: TFLiteConversionConfig,
    model_class: Optional[type] = None,
    model_instance = None,
) -> dict:
    """Convert PyTorch model to TFLite.
    
    Args:
        pytorch_model_path: Path to .pth checkpoint
        output_path: Path to save TFLite model
        config: Conversion configuration
        model_class: PyTorch model class (used if model_instance not provided)
        model_instance: Pre-instantiated model (optional, takes precedence)
    
    Returns:
        Dictionary with conversion metadata
    """
    import torch
    
    if not pytorch_model_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pytorch_model_path}")
    
    # Load or use provided model
    if model_instance is not None:
        model = model_instance
    elif model_class is not None:
        model = model_class()
        checkpoint = torch.load(pytorch_model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError("Either model_class or model_instance must be provided")
    
    model.eval()
    
    # Export to ONNX first
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        
        onnx_result = export_pytorch_to_onnx(
            model,
            onnx_path,
            input_shape=config.input_shape,
            opset_version=config.opset_version,
        )
        
        # Create representative data generator if needed
        rep_data_gen = None
        if config.quantization == "int8" and config.representative_dataset_path:
            rep_data = load_representative_dataset(
                config.representative_dataset_path,
                max_samples=100,
            )
            def rep_data_gen():
                for sample in rep_data:
                    yield [sample.astype(np.float32)]
        
        # Convert ONNX to TFLite
        tflite_result = convert_onnx_to_tflite(
            onnx_path,
            output_path,
            config,
            representative_data_gen=rep_data_gen,
        )
    
    return {
        "pytorch_path": str(pytorch_model_path),
        "tflite_path": str(output_path),
        "size_mb": tflite_result["size_mb"],
        "quantization": config.quantization,
        "onnx_export": onnx_result,
    }


def load_representative_dataset(
    dataset_path: Path,
    max_samples: int = 100,
    input_shape: tuple = (1, 3, 256, 256),
) -> np.ndarray:
    """Load representative dataset for int8 quantization.
    
    Args:
        dataset_path: Path to directory with images or .npy file
        max_samples: Maximum number of samples to use
        input_shape: Expected input shape (NCHW)
    
    Returns:
        Numpy array of shape (N, C, H, W) with representative inputs
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Representative dataset not found: {dataset_path}")
    
    # Standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    if dataset_path.suffix == ".npy":
        # Load directly from numpy file
        data = np.load(dataset_path)
        return data[:max_samples]
    
    elif dataset_path.is_dir():
        # Load images from directory
        from PIL import Image
        
        samples = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        for img_path in sorted(dataset_path.iterdir())[:max_samples]:
            if img_path.suffix.lower() not in image_extensions:
                continue
            
            img = Image.open(img_path).convert("RGB")
            img = img.resize((input_shape[3], input_shape[2]), Image.Resampling.LANCZOS)
            
            # To numpy, normalize
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            arr = (arr - mean.squeeze(0)) / std.squeeze(0)
            
            samples.append(arr)
        
        if not samples:
            raise ValueError(f"No valid images found in {dataset_path}")
        
        return np.stack(samples, axis=0)
    
    else:
        raise ValueError(f"dataset_path must be .npy file or directory: {dataset_path}")


def validate_tflite_model(tflite_model_path: Path) -> dict:
    """Validate TFLite model and extract metadata.
    
    Args:
        tflite_model_path: Path to TFLite model
    
    Returns:
        Dictionary with model metadata (input/output shapes, ops, size, etc.)
    """
    if not tflite_model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_model_path}")
    
    size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)
    
    result = {
        "model_path": str(tflite_model_path),
        "size_mb": size_mb,
    }
    
    try:
        import tensorflow as tf
        
        # Load with TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()
        
        # Get details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        result.update({
            "status": "valid",
            "inputs": [
                {
                    "name": d["name"],
                    "shape": d["shape"].tolist(),
                    "dtype": str(d["dtype"]),
                }
                for d in input_details
            ],
            "outputs": [
                {
                    "name": d["name"],
                    "shape": d["shape"].tolist(),
                    "dtype": str(d["dtype"]),
                }
                for d in output_details
            ],
        })
        
    except ImportError:
        result["status"] = "validation_skipped"
        result["note"] = "TensorFlow not installed, cannot validate model"
    except Exception as e:
        result["status"] = "validation_failed"
        result["error"] = str(e)
    
    return result


def run_tflite_inference(
    tflite_model_path: Path,
    input_data: np.ndarray,
) -> list:
    """Run inference with TFLite model.
    
    Args:
        tflite_model_path: Path to TFLite model
        input_data: Input numpy array (NCHW format will be converted to NHWC)
    
    Returns:
        List of output tensors
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required for TFLite inference: pip install tensorflow")
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Handle NCHW -> NHWC conversion if needed
    if len(input_data.shape) == 4:
        # Check if TFLite expects NHWC (common) vs NCHW
        expected_shape = input_details[0]["shape"]
        if expected_shape[-1] == 3:  # NHWC expected
            input_data = input_data.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    
    interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
    interpreter.invoke()
    
    outputs = []
    for detail in output_details:
        outputs.append(interpreter.get_tensor(detail["index"]))
    
    return outputs


if __name__ == "__main__":
    # Test the module
    print("TFLite Converter Module")
    print("=" * 40)
    
    # Test config creation
    config = TFLiteConversionConfig()
    print(f"Default config: {config}")
    
    # Test representative dataset loading (if directory exists)
    test_dir = Path("ECDD_Experiment_Data/real")
    if test_dir.exists():
        try:
            data = load_representative_dataset(test_dir, max_samples=5)
            print(f"Loaded representative data: {data.shape}")
        except Exception as e:
            print(f"Could not load representative data: {e}")
    else:
        print(f"Test directory not found: {test_dir}")
    
    print("\nTo convert a model:")
    print("  from Training.models.ladeda_resnet import LaDeDaResNet50")
    print("  result = convert_pytorch_to_tflite(")
    print("      Path('model.pth'),")
    print("      Path('model.tflite'),")
    print("      TFLiteConversionConfig(),")
    print("      model_class=LaDeDaResNet50,")
    print("  )")
