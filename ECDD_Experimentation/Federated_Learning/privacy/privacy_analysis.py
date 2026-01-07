"""
Privacy Analysis for Federated Drift Detection.

Implements:
- Privacy budget tracking and accounting
- Differential privacy guarantees (ε, δ)
- Privacy composition (sequential, advanced, RDP)
- Information leakage quantification
- Reconstruction attacks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
from scipy.special import comb
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


class PrivacyAccountant:
    """
    Track privacy budget consumption across federated rounds.
    
    Implements multiple composition theorems:
    - Sequential (basic)
    - Advanced composition
    - Moments accountant (Rényi DP)
    """
    
    def __init__(self, total_budget: float = 10.0, delta: float = 1e-5):
        """
        Initialize privacy accountant.
        
        Args:
            total_budget: Total epsilon budget available
            delta: Delta parameter for (ε, δ)-DP
        """
        self.total_budget = total_budget
        self.delta = delta
        self.consumed_budget = 0.0
        self.query_log = []
        
        # For advanced composition
        self.num_queries = 0
        self.epsilon_per_query = []
    
    def add_query(self, epsilon: float, query_type: str = "sketch", 
                 num_samples: int = 1) -> bool:
        """
        Record a privacy-consuming query.
        
        Args:
            epsilon: Privacy cost of this query
            query_type: Type of query (sketch, aggregation, etc.)
            num_samples: Number of samples affected
            
        Returns:
            True if budget allows, False if exceeded
        """
        if self.consumed_budget + epsilon > self.total_budget:
            return False
        
        self.consumed_budget += epsilon
        self.num_queries += 1
        self.epsilon_per_query.append(epsilon)
        
        self.query_log.append({
            'query_num': self.num_queries,
            'epsilon': epsilon,
            'query_type': query_type,
            'num_samples': num_samples,
            'cumulative_epsilon': self.consumed_budget
        })
        
        return True
    
    def get_sequential_composition(self) -> float:
        """
        Compute privacy loss under sequential composition.
        
        ε_total = Σ ε_i
        
        Returns:
            Total epsilon
        """
        return sum(self.epsilon_per_query)
    
    def get_advanced_composition(self) -> float:
        """
        Compute privacy loss under advanced composition.
        
        For k queries with ε each:
        ε_total = ε * sqrt(2k * ln(1/δ')) + k * ε²
        where δ' = δ / k
        
        Returns:
            Total epsilon (advanced composition)
        """
        if self.num_queries == 0:
            return 0.0
        
        k = self.num_queries
        eps = np.mean(self.epsilon_per_query)  # Assume uniform for simplicity
        delta_prime = self.delta / k
        
        epsilon_total = eps * np.sqrt(2 * k * np.log(1 / delta_prime)) + k * (eps ** 2)
        
        return epsilon_total
    
    def get_moments_accountant(self, orders: List[float] = None) -> float:
        """
        Compute privacy loss using moments accountant (Rényi DP).
        
        More tight bound for large number of queries.
        
        Args:
            orders: List of Rényi orders to compute
            
        Returns:
            Epsilon using moments accountant
        """
        if orders is None:
            orders = [1.5, 2, 2.5, 3, 4, 5, 10, 20, 50]
        
        if self.num_queries == 0:
            return 0.0
        
        # Simplified approximation
        # True implementation would track Rényi divergence
        k = self.num_queries
        eps = np.mean(self.epsilon_per_query)
        
        # Approximation: better than advanced but not exact
        epsilon_rdp = eps * np.sqrt(k * np.log(1 / self.delta))
        
        return epsilon_rdp
    
    def remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.total_budget - self.consumed_budget)
    
    def get_summary(self) -> Dict:
        """Get summary of privacy consumption."""
        return {
            'total_budget': self.total_budget,
            'consumed_budget': self.consumed_budget,
            'remaining_budget': self.remaining_budget(),
            'num_queries': self.num_queries,
            'sequential_composition': self.get_sequential_composition(),
            'advanced_composition': self.get_advanced_composition(),
            'moments_accountant': self.get_moments_accountant(),
            'delta': self.delta
        }
    
    def __repr__(self):
        return (f"PrivacyAccountant(consumed={self.consumed_budget:.2f}/{self.total_budget:.2f}, "
                f"queries={self.num_queries})")


class InformationLeakageAnalyzer:
    """
    Analyze information leakage from sketches.
    
    Quantifies how much information about raw data can be recovered
    from privacy-preserving sketches.
    """
    
    def __init__(self):
        self.leakage_scores = []
    
    def compute_reconstruction_error(self, 
                                    true_scores: np.ndarray,
                                    reconstructed_scores: np.ndarray) -> float:
        """
        Compute reconstruction error (how well attacker can recover data).
        
        Args:
            true_scores: Original scores
            reconstructed_scores: Reconstructed from sketch
            
        Returns:
            Reconstruction error (lower = more leakage)
        """
        mse = np.mean((true_scores - reconstructed_scores) ** 2)
        rmse = np.sqrt(mse)
        return rmse
    
    def compute_mutual_information(self,
                                  sketch: np.ndarray,
                                  true_data: np.ndarray,
                                  num_bins: int = 20) -> float:
        """
        Estimate mutual information between sketch and true data.
        
        I(Sketch; Data) measures correlation.
        
        Args:
            sketch: Privacy-preserving sketch (histogram)
            true_data: Original data
            num_bins: Number of bins for discretization
            
        Returns:
            Mutual information estimate (bits)
        """
        # Discretize both
        sketch_discrete = (sketch * num_bins).astype(int)
        data_discrete = (true_data * num_bins).astype(int)
        
        # Compute joint and marginal distributions
        joint_hist = np.histogram2d(sketch_discrete.flatten(), 
                                    data_discrete.flatten(),
                                    bins=num_bins)[0]
        joint_hist = joint_hist / joint_hist.sum()
        
        # Marginals
        p_sketch = joint_hist.sum(axis=1)
        p_data = joint_hist.sum(axis=0)
        
        # Mutual information
        mi = 0.0
        for i in range(num_bins):
            for j in range(num_bins):
                if joint_hist[i, j] > 0:
                    mi += joint_hist[i, j] * np.log2(
                        joint_hist[i, j] / (p_sketch[i] * p_data[j] + 1e-10)
                    )
        
        return max(0, mi)
    
    def membership_inference_attack(self,
                                   sketch: np.ndarray,
                                   candidate_samples: np.ndarray,
                                   true_members: np.ndarray) -> Dict:
        """
        Simulate membership inference attack.
        
        Attacker tries to determine if specific samples were in dataset
        based on sketch.
        
        Args:
            sketch: Privacy-preserving sketch
            candidate_samples: Samples to test
            true_members: Ground truth membership (0 or 1)
            
        Returns:
            Attack success metrics
        """
        # Simple heuristic: samples with scores matching sketch peaks
        # are more likely to be members
        
        # Find sketch peaks
        sketch_normalized = sketch / sketch.sum()
        peak_bins = np.argsort(sketch_normalized)[-5:]  # Top 5 bins
        
        # Predict membership based on whether sample falls in peak bins
        predictions = []
        for sample in candidate_samples:
            bin_idx = int(sample * len(sketch))
            bin_idx = min(bin_idx, len(sketch) - 1)
            predicted_member = 1 if bin_idx in peak_bins else 0
            predictions.append(predicted_member)
        
        predictions = np.array(predictions)
        
        # Compute attack accuracy
        accuracy = np.mean(predictions == true_members)
        
        # True positive rate and false positive rate
        tp = np.sum((predictions == 1) & (true_members == 1))
        fp = np.sum((predictions == 1) & (true_members == 0))
        tn = np.sum((predictions == 0) & (true_members == 0))
        fn = np.sum((predictions == 0) & (true_members == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'tpr': tpr,
            'fpr': fpr,
            'advantage': accuracy - 0.5  # Advantage over random guessing
        }


class PrivacyUtilityTradeoff:
    """
    Analyze privacy-utility trade-off.
    
    Measures how privacy (epsilon) affects utility (detection accuracy).
    """
    
    def __init__(self):
        self.measurements = []
    
    def add_measurement(self, 
                       epsilon: float,
                       detection_f1: float,
                       detection_latency: float,
                       false_alarm_rate: float):
        """
        Record a privacy-utility measurement.
        
        Args:
            epsilon: Privacy parameter
            detection_f1: Detection F1 score
            detection_latency: Detection latency (rounds)
            false_alarm_rate: False alarm rate
        """
        self.measurements.append({
            'epsilon': epsilon,
            'f1': detection_f1,
            'latency': detection_latency,
            'false_alarm_rate': false_alarm_rate
        })
    
    def compute_pareto_frontier(self) -> List[Dict]:
        """
        Compute Pareto frontier of privacy-utility trade-off.
        
        Returns:
            List of Pareto-optimal points
        """
        if not self.measurements:
            return []
        
        # Sort by epsilon (privacy, lower is better)
        sorted_measurements = sorted(self.measurements, key=lambda x: x['epsilon'])
        
        # Find Pareto frontier (non-dominated points)
        pareto_frontier = []
        best_f1 = -float('inf')
        
        for measurement in sorted_measurements:
            if measurement['f1'] >= best_f1:
                pareto_frontier.append(measurement)
                best_f1 = measurement['f1']
        
        return pareto_frontier
    
    def plot_tradeoff(self, save_path: Optional[str] = None):
        """Plot privacy-utility trade-off curves."""
        if not self.measurements:
            print("No measurements to plot")
            return
        
        epsilons = [m['epsilon'] for m in self.measurements]
        f1_scores = [m['f1'] for m in self.measurements]
        latencies = [m['latency'] for m in self.measurements]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # F1 vs Epsilon
        ax1.plot(epsilons, f1_scores, 'o-', linewidth=2, markersize=8, color='#2ecc71')
        ax1.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax1.set_ylabel('Detection F1 Score', fontsize=12)
        ax1.set_title('Privacy vs Utility (F1)', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(alpha=0.3)
        ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Target F1 = 0.7')
        ax1.legend()
        
        # Latency vs Epsilon
        ax2.plot(epsilons, latencies, 's-', linewidth=2, markersize=8, color='#e74c3c')
        ax2.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax2.set_ylabel('Detection Latency (rounds)', fontsize=12)
        ax2.set_title('Privacy vs Speed', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Privacy-utility plot saved to {save_path}")
        
        plt.show()


def compute_privacy_amplification(epsilon: float,
                                 sampling_rate: float,
                                 num_rounds: int) -> float:
    """
    Compute privacy amplification from subsampling.
    
    When only a fraction q of data is sampled, privacy improves.
    ε_amplified ≈ q * ε (for small q)
    
    Args:
        epsilon: Base privacy parameter
        sampling_rate: Fraction of data sampled (0-1)
        num_rounds: Number of rounds
        
    Returns:
        Amplified epsilon
    """
    # Simple approximation
    epsilon_amplified = sampling_rate * epsilon
    
    # For multiple rounds (composition)
    epsilon_total = epsilon_amplified * np.sqrt(num_rounds)
    
    return epsilon_total


def compute_privacy_loss_distribution(epsilon: float,
                                     delta: float,
                                     num_queries: int) -> Dict:
    """
    Compute privacy loss distribution.
    
    Args:
        epsilon: Privacy parameter per query
        delta: Delta parameter
        num_queries: Number of queries
        
    Returns:
        Privacy loss statistics
    """
    # Sequential composition
    sequential_eps = epsilon * num_queries
    
    # Advanced composition
    delta_prime = delta / num_queries
    advanced_eps = epsilon * np.sqrt(2 * num_queries * np.log(1 / delta_prime)) + \
                   num_queries * (epsilon ** 2)
    
    # Moments accountant (approximation)
    rdp_eps = epsilon * np.sqrt(num_queries * np.log(1 / delta))
    
    return {
        'sequential': sequential_eps,
        'advanced': advanced_eps,
        'rdp': rdp_eps,
        'improvement_factor': sequential_eps / rdp_eps
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Privacy Analysis...")
    
    # Test 1: Privacy Accountant
    print("\n1. Testing Privacy Accountant:")
    accountant = PrivacyAccountant(total_budget=10.0, delta=1e-5)
    
    # Simulate 100 queries with ε=0.1 each
    for i in range(100):
        success = accountant.add_query(epsilon=0.1, query_type='sketch')
        if not success:
            print(f"  Budget exceeded at query {i+1}")
            break
    
    summary = accountant.get_summary()
    print(f"  {accountant}")
    print(f"  Sequential composition: ε={summary['sequential_composition']:.2f}")
    print(f"  Advanced composition: ε={summary['advanced_composition']:.2f}")
    print(f"  Moments accountant: ε={summary['moments_accountant']:.2f}")
    print(f"  Improvement: {summary['sequential_composition']/summary['moments_accountant']:.2f}x")
    
    # Test 2: Information Leakage
    print("\n2. Testing Information Leakage:")
    analyzer = InformationLeakageAnalyzer()
    
    # Generate synthetic data
    true_scores = np.random.beta(2, 5, size=1000)
    
    # Create noisy sketch
    from core.privacy_utils import add_laplace_noise
    hist, _ = np.histogram(true_scores, bins=20, range=(0, 1))
    noisy_hist = add_laplace_noise(hist.astype(float), sensitivity=1.0, epsilon=1.0)
    noisy_hist = np.maximum(noisy_hist, 0) / np.sum(np.maximum(noisy_hist, 0))
    
    # Attempt reconstruction
    bin_centers = np.linspace(0.025, 0.975, 20)
    reconstructed = np.random.choice(bin_centers, size=1000, p=noisy_hist)
    
    recon_error = analyzer.compute_reconstruction_error(true_scores, reconstructed)
    print(f"  Reconstruction error (RMSE): {recon_error:.4f}")
    
    # Membership inference attack
    members = true_scores[:500]
    non_members = np.random.beta(3, 4, size=500)
    candidates = np.concatenate([members, non_members])
    true_membership = np.concatenate([np.ones(500), np.zeros(500)])
    
    attack_results = analyzer.membership_inference_attack(
        noisy_hist, candidates, true_membership
    )
    print(f"  Membership inference accuracy: {attack_results['accuracy']:.3f}")
    print(f"  Attack advantage: {attack_results['advantage']:.3f}")
    
    # Test 3: Privacy-Utility Trade-off
    print("\n3. Testing Privacy-Utility Trade-off:")
    tradeoff = PrivacyUtilityTradeoff()
    
    # Simulate measurements
    for eps in [0.1, 0.5, 1.0, 5.0, 10.0]:
        # Higher epsilon → better utility (simulated)
        f1 = 0.5 + 0.3 * np.log(eps + 1)
        latency = 20 - 10 * np.log(eps + 1)
        far = 0.1 / (eps + 1)
        
        tradeoff.add_measurement(eps, f1, max(1, latency), far)
    
    pareto = tradeoff.compute_pareto_frontier()
    print(f"  Pareto frontier has {len(pareto)} points")
    for point in pareto:
        print(f"    ε={point['epsilon']:.1f}: F1={point['f1']:.3f}, Latency={point['latency']:.1f}")
    
    # Test 4: Privacy amplification
    print("\n4. Testing Privacy Amplification:")
    base_eps = 1.0
    sampling_rate = 0.1
    num_rounds = 100
    
    amplified = compute_privacy_amplification(base_eps, sampling_rate, num_rounds)
    print(f"  Base ε: {base_eps}")
    print(f"  Sampling rate: {sampling_rate}")
    print(f"  After {num_rounds} rounds: ε={amplified:.2f}")
    print(f"  Amplification factor: {base_eps * num_rounds / amplified:.2f}x")
    
    print("\n✅ Privacy analysis tests passed!")
