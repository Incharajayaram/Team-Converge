"""
Experiment 1: Baseline (No Drift)

Objective: Establish baseline performance without any drift.
Measures false alarm rate and communication overhead.

Expected Results:
- Drift detection: False
- False alarm rate: < 5%
- Communication overhead: Baseline reference
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.fed_drift_simulator import HierarchicalFedSimulator
from simulation.evaluation_metrics import ExperimentMetrics, plot_drift_timeline
import matplotlib.pyplot as plt


def run_baseline_experiment(student_model: torch.nn.Module,
                            teacher_model: torch.nn.Module,
                            dataset,
                            baseline_hist: np.ndarray,
                            num_runs: int = 10,
                            save_dir: str = 'results/exp1_baseline'):
    """
    Run baseline experiment (no drift scenario).
    
    Args:
        student_model: Student model
        teacher_model: Teacher model
        dataset: Test dataset
        baseline_hist: Baseline distribution
        num_runs: Number of experiment runs
        save_dir: Directory to save results
        
    Returns:
        Dictionary with aggregated results
    """
    print("=" * 70)
    print("EXPERIMENT 1: BASELINE (NO DRIFT)")
    print("=" * 70)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = {
        'experiment': 'baseline',
        'num_runs': num_runs,
        'num_rounds': 100,
        'num_hubs': 3,
        'students_per_hub': 5,
        'total_students': 15,
        'drift_scenario': None,
        'dropout_rate': 0.2,
        'predictions_per_round': 10
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run multiple times with different seeds
    all_results = []
    
    for run_id in range(num_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_id + 1}/{num_runs}")
        print(f"{'='*70}")
        
        # Set random seed
        random_seed = 42 + run_id
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Create simulator
        simulator = HierarchicalFedSimulator(
            student_model=student_model,
            teacher_model=teacher_model,
            dataset=dataset,
            baseline_hist=baseline_hist,
            num_hubs=config['num_hubs'],
            students_per_hub=config['students_per_hub'],
            non_iid=True,
            powerlaw=True,
            dropout_rate=config['dropout_rate'],
            random_seed=random_seed
        )
        
        # Run experiment (no drift scenario)
        results = simulator.run_experiment(
            num_rounds=config['num_rounds'],
            scenario=None,  # NO DRIFT
            predictions_per_round=config['predictions_per_round'],
            verbose=(run_id == 0)  # Verbose only for first run
        )
        
        # Create metrics tracker (no injection)
        metrics = ExperimentMetrics(injection_round=None, affected_clients=None)
        
        # Update metrics from round logs
        for round_log in results['round_logs']:
            metrics.update(
                round_num=round_log['round'],
                drift_detected=round_log['drift_detected'],
                flagged_clients=round_log['anomalous_hubs'],
                bytes_sent=1024  # Approximate sketch size
            )
        
        # Compute final metrics
        final_metrics = metrics.compute_final_metrics(config['num_rounds'])
        
        # Store results
        run_results = {
            'run_id': run_id,
            'random_seed': random_seed,
            'results': results,
            'metrics': final_metrics
        }
        
        all_results.append(run_results)
        
        # Save individual run
        save_path = Path(save_dir) / f'run_{run_id}.json'
        with open(save_path, 'w') as f:
            json.dump(run_results, f, indent=2, default=str)
        
        print(f"\n  Run {run_id + 1} Results:")
        print(f"    Drift detected: {results['drift_detected']}")
        print(f"    False alarms: {final_metrics['false_alarms']}")
        print(f"    False alarm rate: {final_metrics['false_alarm_rate']:.3f}")
        print(f"    Total predictions: {results['total_predictions']}")
    
    # Aggregate results across runs
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")
    
    aggregated = aggregate_results(all_results)
    
    # Print summary
    print(f"\nFalse Alarm Rate:")
    print(f"  Mean: {aggregated['false_alarm_rate_mean']:.3f}")
    print(f"  Std: {aggregated['false_alarm_rate_std']:.3f}")
    print(f"  Min: {aggregated['false_alarm_rate_min']:.3f}")
    print(f"  Max: {aggregated['false_alarm_rate_max']:.3f}")
    
    print(f"\nCommunication Overhead:")
    print(f"  Mean total bytes: {aggregated['total_bytes_mean']:.0f}")
    print(f"  Std: {aggregated['total_bytes_std']:.0f}")
    
    print(f"\nDrift Detection:")
    print(f"  Runs with drift detected: {aggregated['num_false_positives']}/{num_runs}")
    
    # Save aggregated results
    aggregated['config'] = config
    save_path = Path(save_dir) / 'aggregated_results.json'
    with open(save_path, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {save_dir}/")
    
    return aggregated


def aggregate_results(all_results):
    """Aggregate results from multiple runs."""
    false_alarm_rates = []
    total_bytes = []
    false_positives = 0
    
    for run_results in all_results:
        metrics = run_results['metrics']
        false_alarm_rates.append(metrics['false_alarm_rate'])
        total_bytes.append(metrics['total_bytes'])
        
        if run_results['results']['drift_detected']:
            false_positives += 1
    
    return {
        'num_runs': len(all_results),
        'false_alarm_rate_mean': np.mean(false_alarm_rates),
        'false_alarm_rate_std': np.std(false_alarm_rates),
        'false_alarm_rate_min': np.min(false_alarm_rates),
        'false_alarm_rate_max': np.max(false_alarm_rates),
        'total_bytes_mean': np.mean(total_bytes),
        'total_bytes_std': np.std(total_bytes),
        'num_false_positives': false_positives,
        'false_positive_rate': false_positives / len(all_results)
    }


# Example usage
if __name__ == "__main__":
    print("Experiment 1: Baseline")
    print("This is a template. Replace with your actual models and dataset.")
    
    print("\nTo run:")
    print("  1. Load your trained models")
    print("  2. Load your test dataset")
    print("  3. Compute baseline histogram")
    print("  4. Call run_baseline_experiment()")
    
    print("\nExample code:")
    print("""
    from deepfake_patch_audit.models.student.tiny_ladeda import TinyLaDeDa
    from deepfake_patch_audit.models.teacher.ladeda_wrapper import LaDeDaWrapper
    from deepfake_patch_audit.datasets.celebdf_dataset import CelebDFDataset
    
    # Load models
    student = TinyLaDeDa(...)
    student.load_state_dict(torch.load('outputs/checkpoints_two_stage/student_final.pt'))
    
    teacher = LaDeDaWrapper(...)
    teacher.load_state_dict(torch.load('outputs/checkpoints_teacher/teacher_finetuned_best.pth'))
    
    # Load dataset
    dataset = CelebDFDataset(root='...', split='test')
    
    # Compute baseline
    baseline_hist = compute_baseline_from_validation_set(teacher, val_loader)
    
    # Run experiment
    results = run_baseline_experiment(student, teacher, dataset, baseline_hist)
    """)
