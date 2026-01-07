"""
Master script to run all experiments.

Runs all 5 experiments in sequence:
1. Baseline (no drift)
2. Sudden attack
3. Gradual shift
4. Privacy trade-off
5. Scalability

Usage:
    python run_all_experiments.py --models_path <path> --dataset_path <path>
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import experiments
from experiments.exp1_baseline import run_baseline_experiment
from experiments.exp2_sudden_attack import run_sudden_attack_experiment
from experiments.exp3_gradual_shift import run_gradual_shift_experiment
from experiments.exp4_privacy_tradeoff import run_privacy_tradeoff_experiment
from experiments.exp5_scalability import run_scalability_experiment


def load_models(student_path: str, teacher_path: str):
    """
    Load student and teacher models.
    
    Args:
        student_path: Path to student model checkpoint
        teacher_path: Path to teacher model checkpoint
        
    Returns:
        (student_model, teacher_model)
    """
    print("Loading models...")
    
    # TODO: Replace with actual model loading
    # Example:
    # from deepfake_patch_audit.models.student.tiny_ladeda import TinyLaDeDa
    # from deepfake_patch_audit.models.teacher.ladeda_wrapper import LaDeDaWrapper
    
    # student = TinyLaDeDa(...)
    # student.load_state_dict(torch.load(student_path))
    # student.eval()
    
    # teacher = LaDeDaWrapper(...)
    # teacher.load_state_dict(torch.load(teacher_path))
    # teacher.eval()
    
    # For now, return mock models
    from torch import nn
    
    class MockStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 1)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    class MockTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 1)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    student = MockStudent()
    teacher = MockTeacher()
    
    print("✓ Models loaded")
    return student, teacher


def load_dataset(dataset_path: str):
    """
    Load test dataset.
    
    Args:
        dataset_path: Path to dataset
        
    Returns:
        Dataset
    """
    print("Loading dataset...")
    
    # TODO: Replace with actual dataset loading
    # Example:
    # from deepfake_patch_audit.datasets.celebdf_dataset import CelebDFDataset
    # dataset = CelebDFDataset(root=dataset_path, split='test')
    
    # For now, return mock dataset
    from torch.utils.data import TensorDataset
    
    images = torch.randn(1000, 3, 64, 64)
    labels = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(images, labels)
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    return dataset


def compute_baseline_histogram(model, val_loader=None):
    """
    Compute baseline distribution from validation set.
    
    Args:
        model: Model to use for inference
        val_loader: Validation data loader
        
    Returns:
        Baseline histogram (normalized)
    """
    print("Computing baseline histogram...")
    
    # TODO: Replace with actual baseline computation
    # Example:
    # val_scores = []
    # model.eval()
    # with torch.no_grad():
    #     for images, labels in val_loader:
    #         outputs = model(images)
    #         scores = torch.sigmoid(outputs)
    #         val_scores.extend(scores.cpu().numpy())
    
    # For now, use synthetic baseline
    val_scores = np.random.beta(2, 5, size=5000)
    
    baseline_hist, _ = np.histogram(val_scores, bins=20, range=(0, 1))
    baseline_hist = baseline_hist / baseline_hist.sum()
    
    print(f"✓ Baseline histogram computed (mean={np.mean(val_scores):.3f})")
    return baseline_hist


def run_all_experiments(student_model, teacher_model, dataset, baseline_hist,
                       save_dir: str = 'results',
                       quick_mode: bool = False):
    """
    Run all 5 experiments.
    
    Args:
        student_model: Student model
        teacher_model: Teacher model
        dataset: Test dataset
        baseline_hist: Baseline distribution
        save_dir: Root directory for results
        quick_mode: If True, run with reduced parameters for testing
        
    Returns:
        Dictionary with all experiment results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(save_dir) / f"experiments_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Save directory: {save_dir}")
    print(f"Quick mode: {quick_mode}")
    
    # Adjust parameters for quick mode
    if quick_mode:
        num_runs = 2
        num_runs_per_epsilon = 2
        num_runs_per_scale = 2
        client_counts = [10, 20]
        print("\n⚠️  Quick mode: Reduced runs for testing")
    else:
        num_runs = 10
        num_runs_per_epsilon = 5
        num_runs_per_scale = 3
        client_counts = [10, 20, 50]
    
    all_results = {}
    
    # Experiment 1: Baseline
    print("\n" + "=" * 70)
    print("EXPERIMENT 1/5: BASELINE")
    print("=" * 70)
    try:
        exp1_results = run_baseline_experiment(
            student_model, teacher_model, dataset, baseline_hist,
            num_runs=num_runs,
            save_dir=str(save_dir / 'exp1_baseline')
        )
        all_results['exp1_baseline'] = exp1_results
        print("✓ Experiment 1 complete")
    except Exception as e:
        print(f"✗ Experiment 1 failed: {e}")
        all_results['exp1_baseline'] = {'error': str(e)}
    
    # Experiment 2: Sudden Attack
    print("\n" + "=" * 70)
    print("EXPERIMENT 2/5: SUDDEN ATTACK")
    print("=" * 70)
    try:
        exp2_results = run_sudden_attack_experiment(
            student_model, teacher_model, dataset, baseline_hist,
            num_runs=num_runs,
            save_dir=str(save_dir / 'exp2_sudden_attack')
        )
        all_results['exp2_sudden_attack'] = exp2_results
        print("✓ Experiment 2 complete")
    except Exception as e:
        print(f"✗ Experiment 2 failed: {e}")
        all_results['exp2_sudden_attack'] = {'error': str(e)}
    
    # Experiment 3: Gradual Shift
    print("\n" + "=" * 70)
    print("EXPERIMENT 3/5: GRADUAL SHIFT")
    print("=" * 70)
    try:
        exp3_results = run_gradual_shift_experiment(
            student_model, teacher_model, dataset, baseline_hist,
            num_runs=num_runs,
            save_dir=str(save_dir / 'exp3_gradual_shift')
        )
        all_results['exp3_gradual_shift'] = exp3_results
        print("✓ Experiment 3 complete")
    except Exception as e:
        print(f"✗ Experiment 3 failed: {e}")
        all_results['exp3_gradual_shift'] = {'error': str(e)}
    
    # Experiment 4: Privacy Trade-off
    print("\n" + "=" * 70)
    print("EXPERIMENT 4/5: PRIVACY TRADE-OFF")
    print("=" * 70)
    try:
        exp4_results = run_privacy_tradeoff_experiment(
            student_model, teacher_model, dataset, baseline_hist,
            epsilon_values=[0.1, 0.5, 1.0, 5.0, 10.0, float('inf')],
            num_runs_per_epsilon=num_runs_per_epsilon,
            save_dir=str(save_dir / 'exp4_privacy_tradeoff')
        )
        all_results['exp4_privacy_tradeoff'] = exp4_results
        print("✓ Experiment 4 complete")
    except Exception as e:
        print(f"✗ Experiment 4 failed: {e}")
        all_results['exp4_privacy_tradeoff'] = {'error': str(e)}
    
    # Experiment 5: Scalability
    print("\n" + "=" * 70)
    print("EXPERIMENT 5/5: SCALABILITY")
    print("=" * 70)
    try:
        exp5_results = run_scalability_experiment(
            student_model, teacher_model, dataset, baseline_hist,
            client_counts=client_counts,
            num_runs_per_scale=num_runs_per_scale,
            save_dir=str(save_dir / 'exp5_scalability')
        )
        all_results['exp5_scalability'] = exp5_results
        print("✓ Experiment 5 complete")
    except Exception as e:
        print(f"✗ Experiment 5 failed: {e}")
        all_results['exp5_scalability'] = {'error': str(e)}
    
    # Save master results
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    
    master_results = {
        'timestamp': timestamp,
        'quick_mode': quick_mode,
        'experiments': all_results
    }
    
    save_path = save_dir / 'master_results.json'
    with open(save_path, 'w') as f:
        json.dump(master_results, f, indent=2, default=str)
    
    print(f"\n✓ Master results saved to {save_path}")
    print(f"✓ All experiment results saved to {save_dir}/")
    
    return master_results


def main():
    parser = argparse.ArgumentParser(description='Run all federated drift detection experiments')
    parser.add_argument('--student_model', type=str, default='outputs/checkpoints_two_stage/student_final.pt',
                       help='Path to student model checkpoint')
    parser.add_argument('--teacher_model', type=str, default='outputs/checkpoints_teacher/teacher_finetuned_best.pth',
                       help='Path to teacher model checkpoint')
    parser.add_argument('--dataset', type=str, default='data/celebdf/test',
                       help='Path to test dataset')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced runs')
    
    args = parser.parse_args()
    
    # Load components
    student, teacher = load_models(args.student_model, args.teacher_model)
    dataset = load_dataset(args.dataset)
    baseline_hist = compute_baseline_histogram(teacher)
    
    # Run all experiments
    results = run_all_experiments(
        student, teacher, dataset, baseline_hist,
        save_dir=args.save_dir,
        quick_mode=args.quick
    )
    
    print("\n🎉 All experiments complete!")


if __name__ == "__main__":
    main()
