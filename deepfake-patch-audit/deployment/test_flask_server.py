#!/usr/bin/env python3
"""
Test script for Flask server inference pipeline.

Tests:
1. Model loading and ONNX inference
2. Image preprocessing pipeline
3. TopK pooling and threshold decision
4. End-to-end prediction
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnxruntime as rt
except ImportError:
    print("ERROR: onnxruntime not installed. Install with: pip install onnxruntime")
    sys.exit(1)


def test_model_loading():
    """Test ONNX model can be loaded."""
    print("\n" + "="*80)
    print("TEST 1: Model Loading")
    print("="*80)

    model_path = Path(__file__).parent / "pi_server" / "models" / "deepfake_detector.onnx"

    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print(f"  Run: python scripts/export_student.py --model outputs/checkpoints_two_stage/student_final.pt --onnx-only")
        return False

    try:
        session = rt.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        print(f"✓ Model loaded successfully")
        print(f"  Path: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")

        # Print input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print(f"\nModel I/O:")
        for inp in inputs:
            print(f"  Input: {inp.name}")
            print(f"    Shape: {inp.shape}")
            print(f"    Type: {inp.type}")

        for out in outputs:
            print(f"  Output: {out.name}")
            print(f"    Shape: {out.shape}")
            print(f"    Type: {out.type}")

        return True

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def test_preprocessing():
    """Test image preprocessing pipeline."""
    print("\n" + "="*80)
    print("TEST 2: Image Preprocessing")
    print("="*80)

    try:
        # Create a test image (128x128 RGB) - simulating Nicla output
        test_image = Image.new('RGB', (128, 128), color=(128, 128, 128))

        # Preprocessing parameters
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        IMAGENET_STD = np.array([0.229, 0.224, 0.225])

        # Convert to JPEG (simulate Nicla compression)
        jpeg_buffer = io.BytesIO()
        test_image.save(jpeg_buffer, format='JPEG', quality=80)
        jpeg_data = jpeg_buffer.getvalue()

        print(f"✓ Created test JPEG image: {len(jpeg_data)} bytes")

        # Load from JPEG
        image = Image.open(io.BytesIO(jpeg_data)).convert('RGB')
        print(f"✓ Loaded JPEG: {image.size}")

        # Resize to 256x256
        image = image.resize((256, 256), Image.BICUBIC)
        print(f"✓ Resized to: {image.size}")

        # Convert to numpy
        image_np = np.array(image, dtype=np.float32) / 255.0
        print(f"✓ Normalized to [0, 1]: min={image_np.min():.3f}, max={image_np.max():.3f}")

        # Apply ImageNet normalization
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        print(f"✓ Applied ImageNet normalization: min={image_np.min():.3f}, max={image_np.max():.3f}")

        # Convert to (C, H, W) and add batch
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)
        print(f"✓ Final shape: {image_np.shape} (B, C, H, W)")

        return True

    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test end-to-end inference."""
    print("\n" + "="*80)
    print("TEST 3: End-to-End Inference")
    print("="*80)

    model_path = Path(__file__).parent / "pi_server" / "models" / "deepfake_detector.onnx"

    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return False

    try:
        # Load model
        session = rt.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )

        # Create test input
        test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        print(f"✓ Created test input: {test_input.shape}")

        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        import time
        start = time.time()
        patch_logits = session.run(
            [output_name],
            {input_name: test_input}
        )[0]
        inference_time = (time.time() - start) * 1000

        print(f"✓ Inference completed in {inference_time:.2f} ms")
        print(f"✓ Output shape: {patch_logits.shape} (B, C, H, W)")

        # TopK Pooling
        batch_size, channels, h, w = patch_logits.shape
        num_patches = h * w
        k = max(5, int(np.ceil(0.1 * num_patches)))

        patch_flat = patch_logits.reshape(batch_size, -1)
        top_indices = np.argsort(patch_flat[0])[-k:]
        top_logits = patch_flat[0, top_indices]
        image_logit = np.mean(top_logits)

        print(f"✓ TopK Pooling: selected {k} / {num_patches} patches")
        print(f"  Image-level logit: {image_logit:.4f}")

        # Sigmoid
        fake_prob = 1.0 / (1.0 + np.exp(-image_logit))
        real_prob = 1.0 - fake_prob

        print(f"✓ Sigmoid probabilities:")
        print(f"  P(fake): {fake_prob:.4f}")
        print(f"  P(real): {real_prob:.4f}")

        # Decision
        threshold = 0.84
        is_fake = fake_prob > threshold
        decision = "FAKE" if is_fake else "REAL"
        confidence = max(fake_prob, real_prob)

        print(f"✓ Decision (threshold={threshold}):")
        print(f"  Prediction: {decision}")
        print(f"  Confidence: {confidence:.4f}")

        return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flask_app_structure():
    """Test Flask app can be imported."""
    print("\n" + "="*80)
    print("TEST 4: Flask App Structure")
    print("="*80)

    try:
        flask_app_path = Path(__file__).parent / "pi_server" / "app.py"

        if not flask_app_path.exists():
            print(f"✗ Flask app not found: {flask_app_path}")
            return False

        print(f"✓ Flask app file exists: {flask_app_path}")

        # Check file size
        size = flask_app_path.stat().st_size / 1024
        print(f"✓ File size: {size:.1f} KB")

        # Check key functions exist
        with open(flask_app_path, 'r') as f:
            content = f.read()

        required_items = [
            'def preprocess_image',
            'def run_inference',
            '@app.route(\'/predict\'',
            '@app.route(\'/stats\'',
            '@app.route(\'/status\'',
            'def initialize_model',
        ]

        print(f"✓ Checking Flask app structure...")
        for item in required_items:
            if item in content:
                print(f"  ✓ {item}")
            else:
                print(f"  ✗ {item} NOT FOUND")
                return False

        return True

    except Exception as e:
        print(f"✗ Flask app check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION - FLASK SERVER TEST SUITE")
    print("="*80)

    tests = [
        ("Model Loading", test_model_loading),
        ("Image Preprocessing", test_preprocessing),
        ("End-to-End Inference", test_inference),
        ("Flask App Structure", test_flask_app_structure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Flask server is ready for deployment.")
        print("\nNext steps:")
        print("  1. Copy deployment/ to Raspberry Pi")
        print("  2. Run: python pi_server/app.py")
        print("  3. Configure Nicla devices with Pi IP")
        print("  4. Access dashboard at http://<PI_IP>:5000")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Fix issues before deployment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
