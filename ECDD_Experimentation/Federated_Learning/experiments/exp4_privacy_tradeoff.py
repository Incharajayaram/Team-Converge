"""
Experiment 4: Privacy-Utility Trade-off

Objective: Measure trade-off between privacy (epsilon) and detection accuracy.
Tests system with different differential privacy parameters.

Scenarios:
- Epsilon values: [0.1, 0.5, 1.0, 5.0, 10.0, ∞]
- Same sudden attack scenario for each
- Measure detection accuracy vs privacy budget

Expected Results:
- Higher epsilon → Better detection
- ε=1.0 provides good balance
- Clear privacy-utility trade-off curve
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
    ExperimentMetrics, plot_privacy_utility_tradeoff
)


def run_privacy_tradeoff_experiment(student_model: torch.nn.Module,
                                   teacher_model: torch.nn.Module,
                                   dataset,
                                   baseline_hist: np.ndarray,
                                   epsilon_values: list = [0.1, 0.5, 1.0, 5.0, 10.0, float('inf')],
                                   num_runs_per_epsilon: int = 5,
                                   save_dir: str = 'results/exp4_privacy_tradeoff'):
    """
    Run privacy-utility trade-off experiment.
    
    Args:
        student_model: Student model
        teacher_model: Teacher model
        dataset: Test dataset
        baseline_hist: Baseline distribution
        epsilon_values: List of epsilon values to test
        num_runs_per_epsilon: Runs per epsilon value
        save_dir: Directory to save results
        
    Returns:
        Dictionary with results for each epsilon
    """
    print("=" * 70)
    print("EXPERIMENT 4: PRIVACY-UTILITY TRADE-OFF")
    print("=" * 70)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    base_config = {
        'experiment': 'privacy_tradeoff',
        'epsilon_values': epsilon_values,
        'num_runs_per_epsilon': num_runs_per_epsilon,
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
    print(f"  Epsilon values: {epsilon_values}")
    print(f"  Runs per epsilon: {num_runs_per_epsilon}")
    
    epsilon_results = {}
    
    for epsilon in epsilon_values:
        print(f"\n{'='*70}")
        print(f"TESTING EPSILON = {epsilon}")
        print(f"{'='*70}")
        
        epsilon_runs = []
        
        for run_id in range(num_runs_per_epsilon):
            print(f"\n  Run {run_id + 1}/{num_runs_per_epsilon}")
            
            random_seed = 42 + run_id
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            
            # Determine affected students
            num_affected = int(base_config['total_students'] * base_config['affected_ratio'])
            affected_students = list(np.random.choice(
                base_config['total_students'],
                num_affected,
                replace=False
            ))
            
            # Create scenario
            scenario = SuddenAttackScenario(
                attack_type=base_config['attack_type'],
                intensity=base_config['intensity'],
                start_round=base_config['injection_round'],
                affected_clients=affected_students
            )
            
            # Create simulator with specific epsilon
            # Note: Would need to modify simulator to accept epsilon parameter
            # For now, this is a template showing the structure
            simulator = HierarchicalFedSimulator(
                student_model=student_model,
                teacher_model=teacher_model,
                dataset=dataset,
                baseline_hist=baseline_hist,
                num_hubs=base_config['num_hubs'],
                students_per_hub=base_config['students_per_hub'],
                non_iid=True,
                powerlaw=True,
                dropout_rate=base_config['dropout_rate'],
                random_seed=random_seed
            )
            
            # TODO: Set epsilon for all client monitors
            # This would require extending the simulator initialization
            for student in simulator.students:
                student.monitor.epsilon = epsilon
            
            # Run experiment
            results = simulator.run_experiment(
                num_rounds=base_config['num_rounds'],
                scenario=scenario,
                predictions_per_round=base_config['predictions_per_round'],
                verbose=False
            )
            
            # Compute metrics
            metrics = ExperimentMetrics(
                injection_round=base_config['injection_round'],
                affected_clients=affected_students
            )
            
            for round_log in results['round_logs']:
                metrics.update(
                    round_num=round_log['round'],
                    drift_detected=round_log['drift_detected'],
                    flagged_clients=round_log['anomalous_hubs'],
                    bytes_sent=1024
                )
            
            final_metrics = metrics.compute_final_metrics(base_config['num_rounds'])
            
            epsilon_runs.append({
                'run_id': run_id,
                'epsilon': epsilon,
                'metrics': final_metrics,
                'results': results
            })
            
            print(f"    Detected: {results['drift_detected']}, Latency: {final_metrics['detection_latency']}, F1: {final_metrics.get('client_f1', 'N/A')}")
        
        # Aggregate for this epsilon
        epsilon_aggregated = aggregate_epsilon_results(epsilon_runs)
        epsilon_results[epsilon] = epsilon_aggregated
        
        # Save epsilon results
        save_path = Path(save_dir) / f'epsilon_{epsilon}.json'
        with open(save_path, 'w') as f:
            json.dump({
                'epsilon': epsilon,
                'runs': epsilon_runs,
                'aggregated': epsilon_aggregated
            }, f, indent=2, default=str)
        
        print(f"\n  Epsilon {epsilon} Aggregated:")
        print(f"    Detection rate: {epsilon_aggregated['detection_rate']:.1%}")
        print(f"    Mean F1: {epsilon_aggregated['f1_mean']:.3f}")
        print(f"    Mean latency: {epsilon_aggregated['latency_mean']:.1f}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("PRIVACY-UTILITY TRADE-OFF SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Epsilon':<10} {'Det. Rate':<12} {'F1 Score':<12} {'Latency':<10}")
    print("-" * 50)
    for epsilon in epsilon_values:
        agg = epsilon_results[epsilon]
        print(f"{epsilon:<10} {agg['detection_rate']:<12.1%} "
              f"{agg['f1_mean']:<12.3f} {agg['latency_mean']:<10.1f}")
    
    # Plot
    f1_scores = [epsilon_results[eps]['f1_mean'] for eps in epsilon_values if epsilon_results[eps]['f1_mean'] is not None]
    latencies = [epsilon_results[eps]['latency_mean'] for eps in epsilon_values if epsilon_results[eps]['latency_mean'] is not None]
    
    if f1_scores and latencies:
        plot_privacy_utility_tradeoff(
            epsilon_values[:len(f1_scores)],
            f1_scores,
            latencies,
            save_path=Path(save_dir) / 'privacy_utility_tradeoff.png'
        )
    
    # Save all results
    save_path = Path(save_dir) / 'all_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'config': base_config,
            'epsilon_results': epsilon_results
        }, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {save_dir}/")
    
    return epsilon_results


def aggregate_epsilon_results(epsilon_runs):
    """Aggregate results for a specific epsilon."""
    latencies = []
    f1_scores = []
    detected_count = 0
    
    for run in epsilon_runs:
        metrics = run['metrics']
        
        if metrics.get('detected'):
            detected_count += 1
            if metrics.get('detection_latency') is not None:
                latencies.append(metrics['detection_latency'])
        
        if metrics.get('client_f1') is not None:
            f1_scores.append(metrics['client_f1'])
    
    return {
        'num_runs': len(epsilon_runs),
        'num_detected': detected_count,
        'detection_rate': detected_count / len(epsilon_runs),
        'latency_mean': np.mean(latencies) if latencies else None,
        'latency_std': np.std(latencies) if latencies else None,
        'f1_mean': np.mean(f1_scores) if f1_scores else None,
        'f1_std': np.std(f1_scores) if f1_scores else None
    }


if __name__ == "__main__":
    print("Experiment 4: Privacy-Utility Trade-off")
    print("Replace with your actual models and dataset.")
