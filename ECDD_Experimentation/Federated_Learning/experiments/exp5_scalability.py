"""
Experiment 5: Scalability

Objective: Test system scalability with increasing number of clients.
Measures detection performance and overhead as system scales.

Scenarios:
- Client counts: [10, 20, 50, 100]
- Same sudden attack for each
- Measure: latency, accuracy, communication overhead

Expected Results:
- Detection accuracy stable across scales
- Communication overhead scales linearly
- Aggregation time scales sub-linearly
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.fed_drift_simulator import HierarchicalFedSimulator
from simulation.drift_scenarios import SuddenAttackScenario
from simulation.evaluation_metrics import (
    ExperimentMetrics, plot_scalability
)


def run_scalability_experiment(student_model: torch.nn.Module,
                               teacher_model: torch.nn.Module,
                               dataset,
                               baseline_hist: np.ndarray,
                               client_counts: list = [10, 20, 50],  # 100 would be very slow
                               num_runs_per_scale: int = 3,
                               save_dir: str = 'results/exp5_scalability'):
    """
    Run scalability experiment.
    
    Args:
        student_model: Student model
        teacher_model: Teacher model
        dataset: Test dataset
        baseline_hist: Baseline distribution
        client_counts: List of total client counts to test
        num_runs_per_scale: Runs per scale
        save_dir: Directory to save results
        
    Returns:
        Dictionary with results for each scale
    """
    print("=" * 70)
    print("EXPERIMENT 5: SCALABILITY")
    print("=" * 70)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    base_config = {
        'experiment': 'scalability',
        'client_counts': client_counts,
        'num_runs_per_scale': num_runs_per_scale,
        'num_rounds': 50,  # Reduced for faster testing
        'attack_type': 'blur',
        'injection_round': 25,
        'affected_ratio': 0.3,
        'intensity': 0.3,
        'dropout_rate': 0.2,
        'predictions_per_round': 10,
        'students_per_hub': 5  # Fixed ratio
    }
    
    print(f"\nConfiguration:")
    print(f"  Client counts: {client_counts}")
    print(f"  Runs per scale: {num_runs_per_scale}")
    
    scale_results = {}
    
    for num_clients in client_counts:
        print(f"\n{'='*70}")
        print(f"TESTING WITH {num_clients} CLIENTS")
        print(f"{'='*70}")
        
        # Calculate number of hubs (maintain ~5 students per hub)
        num_hubs = max(2, num_clients // base_config['students_per_hub'])
        students_per_hub = num_clients // num_hubs
        
        print(f"  Configuration: {num_hubs} hubs, ~{students_per_hub} students/hub")
        
        scale_runs = []
        
        for run_id in range(num_runs_per_scale):
            print(f"\n  Run {run_id + 1}/{num_runs_per_scale}")
            
            random_seed = 42 + run_id
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            
            # Determine affected students
            num_affected = int(num_clients * base_config['affected_ratio'])
            affected_students = list(np.random.choice(
                num_clients,
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
            
            # Create simulator
            start_time = time.time()
            
            simulator = HierarchicalFedSimulator(
                student_model=student_model,
                teacher_model=teacher_model,
                dataset=dataset,
                baseline_hist=baseline_hist,
                num_hubs=num_hubs,
                students_per_hub=students_per_hub,
                non_iid=True,
                powerlaw=True,
                dropout_rate=base_config['dropout_rate'],
                random_seed=random_seed
            )
            
            setup_time = time.time() - start_time
            
            # Run experiment
            start_time = time.time()
            
            results = simulator.run_experiment(
                num_rounds=base_config['num_rounds'],
                scenario=scenario,
                predictions_per_round=base_config['predictions_per_round'],
                verbose=False
            )
            
            execution_time = time.time() - start_time
            
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
            
            scale_runs.append({
                'run_id': run_id,
                'num_clients': num_clients,
                'num_hubs': num_hubs,
                'metrics': final_metrics,
                'setup_time': setup_time,
                'execution_time': execution_time,
                'total_time': setup_time + execution_time,
                'results': results
            })
            
            print(f"    Detected: {results['drift_detected']}, "
                  f"Latency: {final_metrics['detection_latency']}, "
                  f"Time: {execution_time:.1f}s")
        
        # Aggregate for this scale
        scale_aggregated = aggregate_scale_results(scale_runs)
        scale_results[num_clients] = scale_aggregated
        
        # Save scale results
        save_path = Path(save_dir) / f'clients_{num_clients}.json'
        with open(save_path, 'w') as f:
            json.dump({
                'num_clients': num_clients,
                'runs': scale_runs,
                'aggregated': scale_aggregated
            }, f, indent=2, default=str)
        
        print(f"\n  {num_clients} Clients Aggregated:")
        print(f"    Detection rate: {scale_aggregated['detection_rate']:.1%}")
        print(f"    Mean latency: {scale_aggregated['latency_mean']:.1f} rounds")
        print(f"    Mean execution time: {scale_aggregated['execution_time_mean']:.1f}s")
        print(f"    Communication overhead: {scale_aggregated['total_bytes_mean']:.0f} bytes")
    
    # Final summary
    print(f"\n{'='*70}")
    print("SCALABILITY SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Clients':<10} {'Det. Rate':<12} {'Latency':<12} {'Time (s)':<12} {'Bytes':<15}")
    print("-" * 65)
    for num_clients in client_counts:
        agg = scale_results[num_clients]
        print(f"{num_clients:<10} {agg['detection_rate']:<12.1%} "
              f"{agg['latency_mean']:<12.1f} "
              f"{agg['execution_time_mean']:<12.1f} "
              f"{agg['total_bytes_mean']:<15.0f}")
    
    # Plot scalability
    latencies = [scale_results[n]['latency_mean'] for n in client_counts 
                if scale_results[n]['latency_mean'] is not None]
    comm_overhead = [scale_results[n]['total_bytes_mean'] / 1024 / 1024  # Convert to MB
                    for n in client_counts]
    
    if latencies:
        plot_scalability(
            client_counts[:len(latencies)],
            latencies,
            comm_overhead[:len(latencies)],
            save_path=Path(save_dir) / 'scalability.png'
        )
    
    # Save all results
    save_path = Path(save_dir) / 'all_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'config': base_config,
            'scale_results': scale_results
        }, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {save_dir}/")
    
    return scale_results


def aggregate_scale_results(scale_runs):
    """Aggregate results for a specific scale."""
    latencies = []
    f1_scores = []
    execution_times = []
    total_bytes = []
    detected_count = 0
    
    for run in scale_runs:
        metrics = run['metrics']
        
        if metrics.get('detected'):
            detected_count += 1
            if metrics.get('detection_latency') is not None:
                latencies.append(metrics['detection_latency'])
        
        if metrics.get('client_f1') is not None:
            f1_scores.append(metrics['client_f1'])
        
        execution_times.append(run['execution_time'])
        total_bytes.append(metrics['total_bytes'])
    
    return {
        'num_runs': len(scale_runs),
        'num_clients': scale_runs[0]['num_clients'],
        'num_hubs': scale_runs[0]['num_hubs'],
        'num_detected': detected_count,
        'detection_rate': detected_count / len(scale_runs),
        'latency_mean': np.mean(latencies) if latencies else None,
        'latency_std': np.std(latencies) if latencies else None,
        'f1_mean': np.mean(f1_scores) if f1_scores else None,
        'f1_std': np.std(f1_scores) if f1_scores else None,
        'execution_time_mean': np.mean(execution_times),
        'execution_time_std': np.std(execution_times),
        'total_bytes_mean': np.mean(total_bytes),
        'total_bytes_std': np.std(total_bytes)
    }


if __name__ == "__main__":
    print("Experiment 5: Scalability")
    print("Replace with your actual models and dataset.")
