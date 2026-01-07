"""
Experiment 2: Sudden Attack Emergence

Objective: Test detection of sudden attack injection.
Simulates new deepfake attack appearing abruptly.

Scenario:
- Attack type: Blur (Gaussian, radius 4.0)
- Injection round: 50
- Affected clients: 30% (random)
- Intensity: 30% of samples

Expected Results:
- Drift detection: True
- Detection latency: < 10 rounds
- Client identification F1: > 0.7
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.fed_drift_simulator import HierarchicalFedSimulator
from simulation.drift_scenarios import SuddenAttackScenario
from simulation.evaluation_metrics import (
    ExperimentMetrics, plot_detection_latency, plot_drift_timeline
)
import matplotlib.pyplot as plt


def run_sudden_attack_experiment(student_model: torch.nn.Module,
                                 teacher_model: torch.nn.Module,
                                 dataset,
                                 baseline_hist: np.ndarray,
                                 num_runs: int = 10,
                                 save_dir: str = 'results/exp2_sudden_attack'):
    """
    Run sudden attack experiment.
    
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
    print("EXPERIMENT 2: SUDDEN ATTACK EMERGENCE")
    print("=" * 70)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = {
        'experiment': 'sudden_attack',
        'num_runs': num_runs,
        'num_rounds': 100,
        'num_hubs': 3,
        'students_per_hub': 5,
        'total_students': 15,
        'attack_type': 'blur',
        'injection_round': 50,
        'affected_ratio': 0.3,
        'intensity': 0.3,
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
        
        # Determine affected students
        num_affected = int(config['total_students'] * config['affected_ratio'])
        affected_students = list(np.random.choice(
            config['total_students'], 
            num_affected, 
            replace=False
        ))
        
        print(f"  Affected students: {affected_students}")
        
        # Create drift scenario
        scenario = SuddenAttackScenario(
            attack_type=config['attack_type'],
            intensity=config['intensity'],
            start_round=config['injection_round'],
            affected_clients=affected_students
        )
        
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
        
        # Run experiment
        results = simulator.run_experiment(
            num_rounds=config['num_rounds'],
            scenario=scenario,
            predictions_per_round=config['predictions_per_round'],
            verbose=(run_id == 0)
        )
        
        # Create metrics tracker
        metrics = ExperimentMetrics(
            injection_round=config['injection_round'],
            affected_clients=affected_students
        )
        
        # Update metrics from round logs
        for round_log in results['round_logs']:
            metrics.update(
                round_num=round_log['round'],
                drift_detected=round_log['drift_detected'],
                flagged_clients=round_log['anomalous_hubs'],
                bytes_sent=1024
            )
        
        # Compute final metrics
        final_metrics = metrics.compute_final_metrics(config['num_rounds'])
        
        # Store results
        run_results = {
            'run_id': run_id,
            'random_seed': random_seed,
            'affected_students': affected_students,
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
        print(f"    Detection latency: {final_metrics['detection_latency']} rounds")
        print(f"    Client F1: {final_metrics['client_f1']:.3f}")
        print(f"    False alarms: {final_metrics['false_alarms']}")
        
        # Plot timeline for first run
        if run_id == 0:
            plot_drift_timeline(metrics, save_path=Path(save_dir) / 'timeline.png')
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")
    
    aggregated = aggregate_sudden_attack_results(all_results)
    
    # Print summary
    print(f"\nDetection Performance:")
    print(f"  Successful detections: {aggregated['num_detected']}/{num_runs}")
    print(f"  Detection rate: {aggregated['detection_rate']:.1%}")
    print(f"  Mean latency: {aggregated['latency_mean']:.1f} ± {aggregated['latency_std']:.1f} rounds")
    
    print(f"\nClient Identification:")
    print(f"  Mean F1: {aggregated['f1_mean']:.3f} ± {aggregated['f1_std']:.3f}")
    print(f"  Mean Precision: {aggregated['precision_mean']:.3f}")
    print(f"  Mean Recall: {aggregated['recall_mean']:.3f}")
    
    # Save aggregated results
    aggregated['config'] = config
    save_path = Path(save_dir) / 'aggregated_results.json'
    with open(save_path, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {save_dir}/")
    
    return aggregated


def aggregate_sudden_attack_results(all_results):
    """Aggregate results from sudden attack runs."""
    latencies = []
    f1_scores = []
    precisions = []
    recalls = []
    detected_count = 0
    
    for run_results in all_results:
        metrics = run_results['metrics']
        
        if metrics['detected']:
            detected_count += 1
            latencies.append(metrics['detection_latency'])
        
        if metrics['client_f1'] is not None:
            f1_scores.append(metrics['client_f1'])
            precisions.append(metrics['client_precision'])
            recalls.append(metrics['client_recall'])
    
    return {
        'num_runs': len(all_results),
        'num_detected': detected_count,
        'detection_rate': detected_count / len(all_results),
        'latency_mean': np.mean(latencies) if latencies else None,
        'latency_std': np.std(latencies) if latencies else None,
        'latency_min': np.min(latencies) if latencies else None,
        'latency_max': np.max(latencies) if latencies else None,
        'f1_mean': np.mean(f1_scores) if f1_scores else None,
        'f1_std': np.std(f1_scores) if f1_scores else None,
        'precision_mean': np.mean(precisions) if precisions else None,
        'recall_mean': np.mean(recalls) if recalls else None
    }


# Example usage
if __name__ == "__main__":
    print("Experiment 2: Sudden Attack")
    print("Replace with your actual models and dataset.")
