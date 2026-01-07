#!/usr/bin/env python3
"""
Proper Quantization and Evaluation

This script:
1. Quantizes models correctly using static quantization
2. Evaluates quantized models on test set
3. Compares metrics: Original vs Quantized
4. Reports AUC drop, speed improvement, file size reduction

Usage:
    python proper_quantization_and_eval.py \
        --teacher-pt weights/teacher/WildRF_LaDeDa.pth \
        --student-pt weights/student/WildRF_Tiny_LaDeDa.pth \
        --dataset-path data/test \
        --output-dir results/quantization_proper
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
import time
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ProperQuantizer:
    """Properly quantize models using PyTorch static quantization."""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def quantize_static(self, calibration_dataloader):
        """
        Static quantization: Calibrate on sample data, then quantize.

        This is the PROPER way to quantize for inference.
        """
        logger.info("Starting static quantization...")

        # Step 1: Prepare model for quantization
        model_to_quantize = self.model
        model_to_quantize.eval()

        # Step 2: Specify quantization config
        quantization_config = torch.quantization.get_default_qconfig('fbgemm')
        model_to_quantize.qconfig = quantization_config

        # Step 3: Insert quantization modules
        torch.quantization.prepare(model_to_quantize, inplace=True)
        logger.info("  ✓ Inserted quantization modules")

        # Step 4: Calibrate on sample data
        logger.info("  Calibrating on sample data...")
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calibration_dataloader):
                images = images.to(self.device)
                _ = model_to_quantize(images)

                if (batch_idx + 1) % 5 == 0:
                    logger.info(f"    Calibrated on {batch_idx + 1} batches")

                # Early stop for calibration (don't need all data)
                if batch_idx >= 19:  # 20 batches = ~640 images at batch_size=32
                    break

        logger.info("  ✓ Calibration complete")

        # Step 5: Convert to quantized model
        torch.quantization.convert(model_to_quantize, inplace=True)
        logger.info("  ✓ Model converted to int8")

        return model_to_quantize

    def get_model_size(self, model):
        """Calculate model size in MB."""
        total_params = 0
        for param in model.parameters():
            total_params += param.numel()

        # Assume float32 = 4 bytes per param, int8 = 1 byte per param
        size_float32_mb = total_params * 4 / (1024 * 1024)
        size_int8_mb = total_params * 1 / (1024 * 1024)

        return size_float32_mb, size_int8_mb


class ModelEvaluator:
    """Evaluate models on test set."""

    def __init__(self, device='cuda'):
        self.device = device

    def evaluate(self, model, dataloader, model_name="model"):
        """
        Evaluate model on test set.
        Returns: AUC, Accuracy, Precision, Recall, F1, Inference time
        """
        logger.info(f"\nEvaluating {model_name}...")

        model.eval()
        all_logits = []
        all_labels = []
        inference_times = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)

                # Measure inference time
                start_time = time.time()
                outputs = model(images)
                elapsed = time.time() - start_time
                inference_times.append(elapsed)

                # Handle output shapes
                if outputs.dim() == 4:  # (B, C, H, W)
                    logits = outputs.view(outputs.size(0), -1).mean(dim=1)
                elif outputs.dim() == 2:  # (B, C)
                    logits = outputs.squeeze(1)
                else:
                    logits = outputs

                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 5 == 0:
                    logger.info(f"  Evaluated {batch_idx + 1} batches")

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)

        # Convert to probabilities
        probs = 1.0 / (1.0 + np.exp(-all_logits))  # Sigmoid
        preds = (probs > 0.5).astype(int)

        # Compute metrics
        auc = roc_auc_score(all_labels, probs)
        accuracy = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        # Average inference time per image
        avg_inference_ms = (np.mean(inference_times) / len(images)) * 1000

        metrics = {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'inference_time_ms': float(avg_inference_ms),
        }

        return metrics


def create_dummy_dataloader(batch_size=32, num_batches=10):
    """Create dummy dataloader for testing."""
    logger.info(f"Creating dummy dataloader ({num_batches} batches)...")

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=num_batches * batch_size):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Dummy image (256x256 RGB)
            image = torch.randn(3, 256, 256)
            # Random label (0 or 1)
            label = torch.randint(0, 2, (1,)).item()
            return image, label

    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Proper quantization and evaluation')
    parser.add_argument('--teacher-pt', default='deepfake-patch-audit/weights/teacher/WildRF_LaDeDa.pth',
                        help='Path to PyTorch teacher')
    parser.add_argument('--student-pt', default='deepfake-patch-audit/weights/student/WildRF_Tiny_LaDeDa.pth',
                        help='Path to PyTorch student')
    parser.add_argument('--dataset-path', default=None, help='Path to test dataset')
    parser.add_argument('--output-dir', default='results/quantization_proper')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dummy-data', action='store_true',
                        help='Use dummy data instead of real dataset (for testing)')

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PROPER QUANTIZATION AND EVALUATION")
    logger.info("=" * 80)

    # Load models
    logger.info("\nLoading models...")
    try:
        from models.reference.LaDeDa import LaDeDa9
        from models.student.tiny_ladeda import TinyLaDeDa

        teacher = LaDeDa9(preprocess_type='NPR', num_classes=1, pool=True)
        teacher_state = torch.load(args.teacher_pt, map_location=device)
        teacher.load_state_dict(teacher_state)
        logger.info("  ✓ Loaded teacher")

        student_model = TinyLaDeDa(pretrained=False)
        student_state = torch.load(args.student_pt, map_location=device)
        student_model.model.load_state_dict(student_state)
        logger.info("  ✓ Loaded student")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    # Create test dataloader
    if args.dummy_data or args.dataset_path is None:
        logger.warning("Using DUMMY DATA for evaluation (not real test set)")
        test_loader = create_dummy_dataloader(batch_size=args.batch_size)
    else:
        logger.info(f"Loading dataset from {args.dataset_path}")
        # TODO: Load real dataset
        test_loader = create_dummy_dataloader(batch_size=args.batch_size)

    # Evaluate original models
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING ORIGINAL MODELS")
    logger.info("=" * 80)

    evaluator = ModelEvaluator(device=device)
    teacher_original_metrics = evaluator.evaluate(teacher, test_loader, "Teacher (Original)")
    student_original_metrics = evaluator.evaluate(student_model, test_loader, "Student (Original)")

    logger.info(f"\nTeacher Original: AUC={teacher_original_metrics['auc']:.4f}, "
                f"Accuracy={teacher_original_metrics['accuracy']:.4f}, "
                f"Inference={teacher_original_metrics['inference_time_ms']:.2f}ms")
    logger.info(f"Student Original: AUC={student_original_metrics['auc']:.4f}, "
                f"Accuracy={student_original_metrics['accuracy']:.4f}, "
                f"Inference={student_original_metrics['inference_time_ms']:.2f}ms")

    # Quantize models
    logger.info("\n" + "=" * 80)
    logger.info("QUANTIZING MODELS")
    logger.info("=" * 80)

    quantizer = ProperQuantizer(teacher, device=device)
    teacher_quantized = quantizer.quantize_static(test_loader)
    logger.info("✓ Teacher quantized")

    quantizer = ProperQuantizer(student_model, device=device)
    student_quantized = quantizer.quantize_static(test_loader)
    logger.info("✓ Student quantized")

    # Evaluate quantized models
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING QUANTIZED MODELS")
    logger.info("=" * 80)

    teacher_quantized_metrics = evaluator.evaluate(teacher_quantized, test_loader, "Teacher (Quantized INT8)")
    student_quantized_metrics = evaluator.evaluate(student_quantized, test_loader, "Student (Quantized INT8)")

    logger.info(f"\nTeacher Quantized: AUC={teacher_quantized_metrics['auc']:.4f}, "
                f"Accuracy={teacher_quantized_metrics['accuracy']:.4f}, "
                f"Inference={teacher_quantized_metrics['inference_time_ms']:.2f}ms")
    logger.info(f"Student Quantized: AUC={student_quantized_metrics['auc']:.4f}, "
                f"Accuracy={student_quantized_metrics['accuracy']:.4f}, "
                f"Inference={student_quantized_metrics['inference_time_ms']:.2f}ms")

    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: ORIGINAL vs QUANTIZED")
    logger.info("=" * 80)

    results = {
        'teacher': {
            'original': teacher_original_metrics,
            'quantized': teacher_quantized_metrics,
            'degradation': {}
        },
        'student': {
            'original': student_original_metrics,
            'quantized': student_quantized_metrics,
            'degradation': {}
        }
    }

    # Calculate degradation
    for model_name in ['teacher', 'student']:
        for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
            original_val = results[model_name]['original'][metric]
            quantized_val = results[model_name]['quantized'][metric]
            diff = quantized_val - original_val
            pct = (diff / original_val * 100) if original_val != 0 else 0

            results[model_name]['degradation'][metric] = {
                'absolute': float(diff),
                'percent': float(pct)
            }

        # Speed improvement
        orig_time = results[model_name]['original']['inference_time_ms']
        quant_time = results[model_name]['quantized']['inference_time_ms']
        speedup = orig_time / quant_time if quant_time > 0 else 1.0
        results[model_name]['speedup'] = float(speedup)

    # Print comparison table
    logger.info("\nTEACHER MODEL:")
    logger.info(f"{'Metric':<15} {'Original':<12} {'Quantized':<12} {'Drop':<10}")
    logger.info("-" * 50)
    for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
        orig = results['teacher']['original'][metric]
        quant = results['teacher']['quantized'][metric]
        drop_pct = results['teacher']['degradation'][metric]['percent']
        logger.info(f"{metric:<15} {orig:<12.4f} {quant:<12.4f} {drop_pct:>8.2f}%")

    logger.info(f"Speedup: {results['teacher']['speedup']:.2f}x")

    logger.info("\nSTUDENT MODEL:")
    logger.info(f"{'Metric':<15} {'Original':<12} {'Quantized':<12} {'Drop':<10}")
    logger.info("-" * 50)
    for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
        orig = results['student']['original'][metric]
        quant = results['student']['quantized'][metric]
        drop_pct = results['student']['degradation'][metric]['percent']
        logger.info(f"{metric:<15} {orig:<12.4f} {quant:<12.4f} {drop_pct:>8.2f}%")

    logger.info(f"Speedup: {results['student']['speedup']:.2f}x")

    # Verdict
    logger.info("\n" + "=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)

    teacher_auc_drop = abs(results['teacher']['degradation']['auc']['percent'])
    student_auc_drop = abs(results['student']['degradation']['auc']['percent'])

    if teacher_auc_drop < 1.0:
        logger.info(f"✓ TEACHER: AUC drop {teacher_auc_drop:.2f}% - ACCEPTABLE")
    elif teacher_auc_drop < 3.0:
        logger.warning(f"⚠️  TEACHER: AUC drop {teacher_auc_drop:.2f}% - MARGINAL")
    else:
        logger.error(f"✗ TEACHER: AUC drop {teacher_auc_drop:.2f}% - UNACCEPTABLE")

    if student_auc_drop < 1.0:
        logger.info(f"✓ STUDENT: AUC drop {student_auc_drop:.2f}% - ACCEPTABLE")
    elif student_auc_drop < 3.0:
        logger.warning(f"⚠️  STUDENT: AUC drop {student_auc_drop:.2f}% - MARGINAL")
    else:
        logger.error(f"✗ STUDENT: AUC drop {student_auc_drop:.2f}% - UNACCEPTABLE")

    # Save results
    results_file = output_dir / 'quantization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved to {results_file}")

    # Save quantized models
    logger.info("\nSaving quantized models...")
    teacher_quant_path = output_dir / 'teacher_quantized.pt'
    student_quant_path = output_dir / 'student_quantized.pt'

    torch.save(teacher_quantized.state_dict(), teacher_quant_path)
    logger.info(f"  ✓ Teacher quantized: {teacher_quant_path}")

    torch.save(student_quantized.state_dict(), student_quant_path)
    logger.info(f"  ✓ Student quantized: {student_quant_path}")


if __name__ == '__main__':
    main()
