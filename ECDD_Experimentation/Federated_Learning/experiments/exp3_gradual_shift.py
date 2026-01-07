"""
Experiment 3: Gradual Distribution Shift

Objective: Test detection of gradual drift over time.
Simulates slowly degrading image quality or evolving attacks.

Scenario:
- Attack type: JPEG compression
- Quality degradation: 100 → 30 over 50 rounds
- Affected clients: All clients
- Start round: 0

Expected Results:
- Drift detection: True
- Detection before 50% intensity
- PSI detector effective for gradual changes
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.fed_drift_simulator import HierarchicalFedSimulator
from simulation.drift_scenarios import GradualDriftScenario
from simulation.evaluation_metrics import ExperimentMetrics
import matplotlib.pyplot as plt


def run_gradual_shift_experiment(student_model: torch.nn.Module,
                                 teacher_model: torch.nn.Module,
                                 dataset,
                                 baseline_hist: np.ndarray,
                                 num_runs: int = 10,
                                 save_dir: str = 'results/exp3_gradual_shift'):
    """
    Run gradual shift experiment.
    
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
    print("EXPERIMENT 3: GRADUAL DISTRIBUTION SHIFT")
    print("=" * 70)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    config = {
        'experiment': 'gradual_shift',
        'num_runs': num_runs,
        'num_rounds': 100,
        'num_hubs': 3,
        'students_per_hub': 5,
        'total_students': 15,
        'attack_type': 'jpeg',
        'intensity_start': 0.0,
        'intensity_end': 0.5,
        'drift_duration': 50,
        'start_round': 0,
        'dropout_rate': 0.2,
        'predictions_per_round': 10
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    all_results = []
    
    for run_id in range(num_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_id + 1}/{num_runs}")
        print(f"{'='*70}")
        
        random_seed = 42 + run_id
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Create gradual drift scenario (affects all clients)
        scenario = GradualDriftScenario(
            attack_type=config['attack_type'],
            intensity_start=config['intensity_start'],
            intensity_end=config['intensity_end'],
            start_round=config['start_round'],
            duration=config['drift_duration'],
            affected_clients=None,  # All clients
            total_clients=config['total_students']
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
        
        # Create metrics (drift starts at round 0, but should detect later)
        metrics = ExperimentMetrics(
            injection_round=config['start_round'],
            affected_clients=list(range(config['total_students']))
        )
        
        for round_log in results['round_logs']:
            metrics.update(
                round_num=round_log['round'],
                drift_detected=round_log['drift_detected'],
                flagged_clients=round_log['anomalous_hubs'],
                bytes_sent=1024
            )
        
        final_metrics = metrics.compute_final_metrics(config['num_rounds'])
        
        # Compute at what intensity was drift detected
        intensity_at_detection = None
        if final_metrics['detected'] and final_metrics['detection_latency'] is not None:
            detection_round = final_metrics['detection_latency']
            if detection_round <= config['drift_duration']:
                progress = detection_round / config['drift_duration']
                intensity_at_detection = config['intensity_start'] + \
                    (config['intensity_end'] - config['intensity_start']) * progress
        
        run_results = {
            'run_id': run_id,
            'random_seed': random_seed,
            'results': results,
            'metrics': final_metrics,
            'intensity_at_detection': intensity_at_detection
        }
        
        all_results.append(run_results)
        
        save_path = Path(save_dir) / f'run_{run_id}.json'
        with open(save_path, 'w') as f:
            json.dump(run_results, f, indent=2, default=str)
        
        print(f"\n  Run {run_id + 1} Results:")
        print(f"    Drift detected: {results['drift_detected']}")
        print(f"    Detection round: {final_metrics['detection_latency']}")
        print(f"    Intensity at detection: {intensity_at_detection:.2%}" if intensity_at_detection else "    Not detected")
    
    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")
    
    aggregated = aggregate_gradual_results(all_results, config)
    
    print(f"\nDetection Performance:")
    print(f"  Successful detections: {aggregated['num_detected']}/{num_runs}")
    print(f"  Mean detection round: {aggregated['detection_round_mean']:.1f} ± {aggregated['detection_round_std']:.1f}")
    print(f"  Mean intensity at detection: {aggregated['intensity_at_detection_mean']:.1%}")
    
    aggregated['config'] = config
    save_path = Path(save_dir) / 'aggregated_results.json'
    with open(save_path, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {save_dir}/")
    
    return aggregated


def aggregate_gradual_results(all_results, config):
    """Aggregate gradual shift results."""
    detection_rounds = []
    intensities_at_detection = []
    detected_count = 0
    
    for run_results in all_results:
        metrics = run_results['metrics']
        
        if metrics['detected']:
            detected_count += 1
            detection_rounds.append(metrics['detection_latency'])
            
            if run_results['intensity_at_detection'] is not None:
                intensities_at_detection.append(run_results['intensity_at_detection'])
    
    return {
        'num_runs': len(all_results),
        'num_detected': detected_count,
        'detection_rate': detected_count / len(all_results),
        'detection_round_mean': np.mean(detection_rounds) if detection_rounds else None,
        'detection_round_std': np.std(detection_rounds) if detection_rounds else None,
        'intensity_at_detection_mean': np.mean(intensities_at_detection) if intensities_at_detection else None,
        'intensity_at_detection_std': np.std(intensities_at_detection) if intensities_at_detection else None,
        'detected_before_50_percent': sum(1 for i in intensities_at_detection if i < 0.5) if intensities_at_detection else 0
    }


if __name__ == "__main__":
    print("Experiment 3: Gradual Shift")
    print("Replace with your actual models and dataset.")
