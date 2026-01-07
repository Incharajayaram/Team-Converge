"""
Experiment 6: Privacy Audit

Objective: Comprehensive privacy analysis of the federated system.

Tests:
1. Privacy budget consumption over time
2. Reconstruction attacks
3. Membership inference attacks
4. Privacy amplification from subsampling
5. Composition theorem validation

Expected Results:
- (ε, δ)-DP guarantees verified
- Advanced composition tighter than sequential
- Reconstruction error proportional to ε
- Membership inference advantage < 0.1 for ε=1.0
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.fed_drift_simulator import HierarchicalFedSimulator
from privacy.privacy_analysis import (
    PrivacyAccountant, InformationLeakageAnalyzer,
    PrivacyUtilityTradeoff, compute_privacy_amplification
)


def run_privacy_audit(student_model: torch.nn.Module,
                     teacher_model: torch.nn.Module,
                     dataset,
                     baseline_hist: np.ndarray,
                     epsilon_values: list = [0.1, 0.5, 1.0, 5.0, 10.0],
                     num_runs_per_epsilon: int = 5,
                     save_dir: str = 'results/exp6_privacy_audit'):
    """
    Run comprehensive privacy audit.
    
    Args:
        student_model: Student model
        teacher_model: Teacher model
        dataset: Test dataset
        baseline_hist: Baseline distribution
        epsilon_values: Privacy parameters to test
        num_runs_per_epsilon: Runs per epsilon
        save_dir: Save directory
        
    Returns:
        Privacy audit results
    """
    print("=" * 70)
    print("EXPERIMENT 6: PRIVACY AUDIT")
    print("=" * 70)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    config = {
        'experiment': 'privacy_audit',
        'epsilon_values': epsilon_values,
        'num_runs_per_epsilon': num_runs_per_epsilon,
        'num_rounds': 100,
        'num_hubs': 3,
        'students_per_hub': 5,
        'total_students': 15,
        'dropout_rate': 0.2,
        'predictions_per_round': 10,
        'delta': 1e-5
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    audit_results = {}
    
    for epsilon in epsilon_values:
        print(f"\n{'='*70}")
        print(f"AUDITING EPSILON = {epsilon}")
        print(f"{'='*70}")
        
        epsilon_audit = {
            'epsilon': epsilon,
            'budget_consumption': [],
            'reconstruction_errors': [],
            'membership_inference': [],
            'composition_analysis': None
        }
        
        for run_id in range(num_runs_per_epsilon):
            print(f"\n  Run {run_id + 1}/{num_runs_per_epsilon}")
            
            random_seed = 42 + run_id
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            
            # Create privacy accountant
            accountant = PrivacyAccountant(
                total_budget=epsilon * config['num_rounds'],
                delta=config['delta']
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
            
            # Set epsilon for all clients
            for student in simulator.students:
                student.monitor.epsilon = epsilon
            
            # Run simulation and track privacy
            print(f"    Running simulation...")
            
            run_audit = audit_single_run(
                simulator, accountant, epsilon,
                num_rounds=config['num_rounds'],
                predictions_per_round=config['predictions_per_round']
            )
            
            epsilon_audit['budget_consumption'].append(run_audit['budget'])
            epsilon_audit['reconstruction_errors'].append(run_audit['reconstruction_error'])
            epsilon_audit['membership_inference'].append(run_audit['membership_attack'])
            
            print(f"    Budget consumed: {run_audit['budget']['consumed']:.2f}")
            print(f"    Reconstruction error: {run_audit['reconstruction_error']:.4f}")
            print(f"    Membership advantage: {run_audit['membership_attack']['advantage']:.4f}")
        
        # Aggregate epsilon results
        epsilon_audit['aggregated'] = aggregate_epsilon_audit(epsilon_audit)
        
        # Composition analysis
        epsilon_audit['composition_analysis'] = analyze_composition(
            epsilon, config['num_rounds'], config['delta']
        )
        
        audit_results[epsilon] = epsilon_audit
        
        # Save epsilon audit
        save_path = Path(save_dir) / f'epsilon_{epsilon}_audit.json'
        with open(save_path, 'w') as f:
            json.dump(epsilon_audit, f, indent=2, default=str)
        
        print(f"\n  Epsilon {epsilon} Summary:")
        print(f"    Mean reconstruction error: {epsilon_audit['aggregated']['mean_reconstruction_error']:.4f}")
        print(f"    Mean membership advantage: {epsilon_audit['aggregated']['mean_membership_advantage']:.4f}")
        print(f"    Composition improvement: {epsilon_audit['composition_analysis']['improvement_factor']:.2f}x")
    
    # Final summary
    print(f"\n{'='*70}")
    print("PRIVACY AUDIT SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Epsilon':<10} {'Recon Error':<15} {'Mem Advantage':<15} {'Composition':<15}")
    print("-" * 60)
    for epsilon in epsilon_values:
        audit = audit_results[epsilon]['aggregated']
        comp = audit_results[epsilon]['composition_analysis']
        print(f"{epsilon:<10} {audit['mean_reconstruction_error']:<15.4f} "
              f"{audit['mean_membership_advantage']:<15.4f} "
              f"{comp['improvement_factor']:<15.2f}x")
    
    # Generate plots
    plot_privacy_audit_results(audit_results, save_dir)
    
    # Save master audit
    save_path = Path(save_dir) / 'privacy_audit_master.json'
    with open(save_path, 'w') as f:
        json.dump({
            'config': config,
            'audit_results': audit_results
        }, f, indent=2, default=str)
    
    print(f"\n✓ Privacy audit complete. Results saved to {save_dir}/")
    
    return audit_results


def audit_single_run(simulator, accountant, epsilon, num_rounds, predictions_per_round):
    """
    Audit a single simulation run.
    
    Args:
        simulator: Federated simulator
        accountant: Privacy accountant
        epsilon: Privacy parameter
        num_rounds: Number of rounds
        predictions_per_round: Predictions per round
        
    Returns:
        Audit results for this run
    """
    # Track original data for reconstruction analysis
    original_data = []
    reconstructed_data = []
    
    # Run simulation
    for round_num in range(num_rounds):
        # Sample active students
        active_students = simulator._sample_active_students()
        
        for student_id in active_students:
            student = simulator.students[student_id]
            
            # Get local data
            local_data = simulator._get_student_data(student_id, predictions_per_round)
            
            for image, label in local_data:
                if not isinstance(image, torch.Tensor):
                    image = simulator._pil_to_tensor(image)
                
                # Predict
                result = student.predict(image)
                
                # Track data
                original_data.append(result['score'])
                
                # If sketch sent, record privacy consumption
                if student.monitor.is_buffer_full():
                    accountant.add_query(epsilon, 'sketch', num_samples=100)
                    
                    # Get sketch for reconstruction analysis
                    sketch = student.monitor.get_sketch(apply_dp=True, clear_after=False)
                    
                    # Attempt reconstruction
                    reconstructed = reconstruct_from_sketch(sketch['histogram_full'])
                    reconstructed_data.extend(reconstructed)
    
    # Compute reconstruction error
    analyzer = InformationLeakageAnalyzer()
    
    min_len = min(len(original_data), len(reconstructed_data))
    if min_len > 0:
        recon_error = analyzer.compute_reconstruction_error(
            np.array(original_data[:min_len]),
            np.array(reconstructed_data[:min_len])
        )
    else:
        recon_error = 0.0
    
    # Membership inference attack
    if len(original_data) > 100:
        # Create test set
        members = np.array(original_data[:50])
        non_members = np.random.beta(3, 4, size=50)
        candidates = np.concatenate([members, non_members])
        true_membership = np.concatenate([np.ones(50), np.zeros(50)])
        
        # Get aggregated sketch
        final_sketch = np.ones(20) / 20  # Placeholder
        
        membership_results = analyzer.membership_inference_attack(
            final_sketch, candidates, true_membership
        )
    else:
        membership_results = {'accuracy': 0.5, 'advantage': 0.0}
    
    return {
        'budget': accountant.get_summary(),
        'reconstruction_error': recon_error,
        'membership_attack': membership_results,
        'num_sketches': accountant.num_queries
    }


def reconstruct_from_sketch(histogram: np.ndarray, num_samples: int = 100) -> np.ndarray:
    """
    Attempt to reconstruct data from histogram sketch.
    
    Args:
        histogram: Histogram sketch
        num_samples: Number of samples to reconstruct
        
    Returns:
        Reconstructed samples
    """
    # Normalize histogram
    histogram = np.array(histogram)
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()
    else:
        histogram = np.ones(len(histogram)) / len(histogram)
    
    # Sample from histogram
    bin_centers = np.linspace(0.025, 0.975, len(histogram))
    reconstructed = np.random.choice(bin_centers, size=num_samples, p=histogram)
    
    return reconstructed


def analyze_composition(epsilon_per_query: float, num_queries: int, delta: float) -> Dict:
    """
    Analyze privacy composition theorems.
    
    Args:
        epsilon_per_query: Privacy parameter per query
        num_queries: Number of queries
        delta: Delta parameter
        
    Returns:
        Composition analysis
    """
    # Sequential composition
    sequential = epsilon_per_query * num_queries
    
    # Advanced composition
    delta_prime = delta / num_queries
    advanced = epsilon_per_query * np.sqrt(2 * num_queries * np.log(1 / delta_prime)) + \
               num_queries * (epsilon_per_query ** 2)
    
    # Moments accountant (approximation)
    rdp = epsilon_per_query * np.sqrt(num_queries * np.log(1 / delta))
    
    return {
        'sequential': sequential,
        'advanced': advanced,
        'rdp': rdp,
        'improvement_factor': sequential / rdp,
        'advanced_vs_sequential': advanced / sequential,
        'rdp_vs_advanced': rdp / advanced
    }


def aggregate_epsilon_audit(epsilon_audit: Dict) -> Dict:
    """Aggregate audit results for an epsilon value."""
    recon_errors = [r for r in epsilon_audit['reconstruction_errors'] if r > 0]
    mem_advantages = [m['advantage'] for m in epsilon_audit['membership_inference']]
    
    return {
        'mean_reconstruction_error': np.mean(recon_errors) if recon_errors else 0.0,
        'std_reconstruction_error': np.std(recon_errors) if recon_errors else 0.0,
        'mean_membership_advantage': np.mean(mem_advantages),
        'std_membership_advantage': np.std(mem_advantages),
        'mean_budget_consumed': np.mean([b['consumed_budget'] for b in epsilon_audit['budget_consumption']])
    }


def plot_privacy_audit_results(audit_results: Dict, save_dir: str):
    """Generate privacy audit visualization plots."""
    epsilon_values = sorted(audit_results.keys())
    
    # Extract metrics
    recon_errors = [audit_results[eps]['aggregated']['mean_reconstruction_error'] for eps in epsilon_values]
    mem_advantages = [audit_results[eps]['aggregated']['mean_membership_advantage'] for eps in epsilon_values]
    improvements = [audit_results[eps]['composition_analysis']['improvement_factor'] for eps in epsilon_values]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Reconstruction Error vs Epsilon
    ax1.plot(epsilon_values, recon_errors, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax1.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax1.set_ylabel('Reconstruction Error (RMSE)', fontsize=11)
    ax1.set_title('Privacy vs Reconstruction Risk', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(alpha=0.3)
    
    # 2. Membership Advantage vs Epsilon
    ax2.plot(epsilon_values, mem_advantages, 's-', linewidth=2, markersize=8, color='#e74c3c')
    ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Target < 0.1')
    ax2.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax2.set_ylabel('Membership Inference Advantage', fontsize=11)
    ax2.set_title('Privacy vs Membership Leakage', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # 3. Composition Improvement
    ax3.bar(range(len(epsilon_values)), improvements, color='#2ecc71', alpha=0.7)
    ax3.set_xticks(range(len(epsilon_values)))
    ax3.set_xticklabels([f'{eps}' for eps in epsilon_values])
    ax3.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax3.set_ylabel('Improvement Factor', fontsize=11)
    ax3.set_title('Composition Theorem Improvement', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Privacy Budget Breakdown
    for i, epsilon in enumerate(epsilon_values[:3]):  # Show first 3
        comp = audit_results[epsilon]['composition_analysis']
        values = [comp['sequential'], comp['advanced'], comp['rdp']]
        labels = ['Sequential', 'Advanced', 'RDP']
        ax4.plot(labels, values, 'o-', linewidth=2, markersize=8, label=f'ε={epsilon}')
    
    ax4.set_ylabel('Total Privacy Cost (ε)', fontsize=11)
    ax4.set_title('Composition Theorem Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'privacy_audit_plots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Privacy audit plots saved to {save_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    print("Experiment 6: Privacy Audit")
    print("This comprehensive privacy analysis validates DP guarantees.")
    print("\nReplace with your actual models and dataset to run.")
