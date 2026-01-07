"""
Evaluation metrics for federated drift detection experiments.

Computes metrics for:
- Detection latency (rounds to detect drift)
- Client/hub identification accuracy
- Communication overhead
- Privacy-utility trade-offs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class ExperimentMetrics:
    """
    Track and compute experiment metrics.
    
    Tracks:
    - Detection latency
    - Client identification (precision, recall, F1)
    - False alarm rate
    - Communication overhead
    """
    
    def __init__(self,
                 injection_round: Optional[int] = None,
                 affected_clients: Optional[List[int]] = None):
        """
        Initialize metrics tracker.
        
        Args:
            injection_round: Round when drift was injected (None if no drift)
            affected_clients: List of affected client IDs (for identification metrics)
        """
        self.injection_round = injection_round
        self.affected_clients = set(affected_clients) if affected_clients else set()
        
        # Detection metrics
        self.detection_round = None
        self.false_alarms = 0
        self.drift_detected = False
        
        # Client identification
        self.identified_clients = set()
        
        # Communication tracking
        self.total_bytes_sent = 0
        self.total_messages = 0
        
        # Round-by-round tracking
        self.round_metrics = []
    
    def update(self,
               round_num: int,
               drift_detected: bool,
               flagged_clients: List[int],
               bytes_sent: Optional[int] = None):
        """
        Update metrics for current round.
        
        Args:
            round_num: Current round number
            drift_detected: Whether drift was detected
            flagged_clients: List of clients flagged as anomalous
            bytes_sent: Bytes sent this round
        """
        # Detection latency
        if drift_detected and self.detection_round is None:
            self.detection_round = round_num
            self.drift_detected = True
        
        # False alarms (drift detected before injection)
        if self.injection_round is not None:
            if drift_detected and round_num < self.injection_round:
                self.false_alarms += 1
        
        # Client identification
        if drift_detected and flagged_clients:
            self.identified_clients.update(flagged_clients)
        
        # Communication
        if bytes_sent:
            self.total_bytes_sent += bytes_sent
            self.total_messages += 1
        
        # Store round metrics
        self.round_metrics.append({
            'round': round_num,
            'drift_detected': drift_detected,
            'flagged_clients': flagged_clients.copy() if flagged_clients else []
        })
    
    def compute_final_metrics(self, total_rounds: int) -> Dict:
        """
        Compute final metrics after experiment.
        
        Args:
            total_rounds: Total number of rounds
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Detection latency
        if self.injection_round is not None:
            if self.detection_round is not None:
                latency = self.detection_round - self.injection_round
                metrics['detection_latency'] = max(0, latency)  # Clamp to 0
                metrics['detected'] = True
            else:
                metrics['detection_latency'] = total_rounds  # Failed to detect
                metrics['detected'] = False
        else:
            metrics['detection_latency'] = None
            metrics['detected'] = None
        
        # False alarm rate
        metrics['false_alarms'] = self.false_alarms
        metrics['false_alarm_rate'] = self.false_alarms / total_rounds if total_rounds > 0 else 0
        
        # Client identification metrics
        if self.affected_clients:
            tp = len(self.identified_clients & self.affected_clients)
            fp = len(self.identified_clients - self.affected_clients)
            fn = len(self.affected_clients - self.identified_clients)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['client_precision'] = precision
            metrics['client_recall'] = recall
            metrics['client_f1'] = f1
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
        else:
            metrics['client_precision'] = None
            metrics['client_recall'] = None
            metrics['client_f1'] = None
        
        # Communication metrics
        metrics['total_bytes'] = self.total_bytes_sent
        metrics['total_messages'] = self.total_messages
        metrics['avg_bytes_per_message'] = (
            self.total_bytes_sent / self.total_messages if self.total_messages > 0 else 0
        )
        
        return metrics
    
    def get_detection_timeline(self) -> Dict:
        """Get timeline of drift detection over rounds."""
        rounds = [m['round'] for m in self.round_metrics]
        detected = [m['drift_detected'] for m in self.round_metrics]
        
        return {
            'rounds': rounds,
            'drift_detected': detected,
            'injection_round': self.injection_round,
            'detection_round': self.detection_round
        }


def compare_methods(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple methods (federated, centralized, isolated).
    
    Args:
        results_dict: Dictionary of {method_name: results}
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison = []
    
    for method_name, results in results_dict.items():
        metrics = results.get('metrics', {})
        
        comparison.append({
            'Method': method_name,
            'Detection Latency': metrics.get('detection_latency', np.nan),
            'Detected': metrics.get('detected', False),
            'Client F1': metrics.get('client_f1', np.nan),
            'False Alarm Rate': metrics.get('false_alarm_rate', np.nan),
            'Total Bytes': metrics.get('total_bytes', 0),
            'Avg Bytes/Message': metrics.get('avg_bytes_per_message', 0)
        })
    
    return pd.DataFrame(comparison)


def plot_detection_latency(results_dict: Dict[str, Dict],
                          save_path: Optional[str] = None):
    """
    Plot detection latency comparison.
    
    Args:
        results_dict: Dictionary of {method_name: results}
        save_path: Optional path to save figure
    """
    methods = []
    latencies = []
    
    for method_name, results in results_dict.items():
        metrics = results.get('metrics', {})
        latency = metrics.get('detection_latency')
        
        if latency is not None and metrics.get('detected', False):
            methods.append(method_name)
            latencies.append(latency)
    
    if not methods:
        print("No detection latency data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, latencies, color=['#2ecc71', '#3498db', '#e74c3c'][:len(methods)])
    
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Detection Latency (rounds)', fontsize=12)
    plt.title('Drift Detection Latency Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(latency)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved detection latency plot to {save_path}")
    
    plt.show()


def plot_privacy_utility_tradeoff(epsilon_values: List[float],
                                  f1_scores: List[float],
                                  latencies: List[float],
                                  save_path: Optional[str] = None):
    """
    Plot privacy-utility trade-off.
    
    Args:
        epsilon_values: List of epsilon values
        f1_scores: Corresponding F1 scores
        latencies: Corresponding detection latencies
        save_path: Optional save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 vs Epsilon
    ax1.plot(epsilon_values, f1_scores, marker='o', linewidth=2, markersize=8,
            color='#2ecc71', label='F1 Score')
    ax1.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax1.set_ylabel('Client Identification F1', fontsize=12)
    ax1.set_title('Privacy vs Utility (F1 Score)', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    
    # Latency vs Epsilon
    ax2.plot(epsilon_values, latencies, marker='s', linewidth=2, markersize=8,
            color='#e74c3c', label='Latency')
    ax2.set_xlabel('Privacy Budget (ε)', fontsize=12)
    ax2.set_ylabel('Detection Latency (rounds)', fontsize=12)
    ax2.set_title('Privacy vs Detection Speed', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved privacy-utility plot to {save_path}")
    
    plt.show()


def plot_scalability(num_clients_list: List[int],
                    latencies: List[float],
                    comm_overhead: List[float],
                    save_path: Optional[str] = None):
    """
    Plot scalability metrics.
    
    Args:
        num_clients_list: List of client counts
        latencies: Detection latencies
        comm_overhead: Communication overhead (bytes)
        save_path: Optional save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Detection latency vs scale
    ax1.plot(num_clients_list, latencies, marker='o', linewidth=2, markersize=8,
            color='#3498db')
    ax1.set_xlabel('Number of Clients', fontsize=12)
    ax1.set_ylabel('Detection Latency (rounds)', fontsize=12)
    ax1.set_title('Scalability: Detection Speed', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Communication overhead vs scale
    ax2.plot(num_clients_list, comm_overhead, marker='s', linewidth=2, markersize=8,
            color='#9b59b6')
    ax2.set_xlabel('Number of Clients', fontsize=12)
    ax2.set_ylabel('Communication Overhead (MB)', fontsize=12)
    ax2.set_title('Scalability: Communication Cost', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scalability plot to {save_path}")
    
    plt.show()


def plot_drift_timeline(metrics: ExperimentMetrics,
                       save_path: Optional[str] = None):
    """
    Plot drift detection timeline.
    
    Args:
        metrics: ExperimentMetrics instance
        save_path: Optional save path
    """
    timeline = metrics.get_detection_timeline()
    
    plt.figure(figsize=(12, 4))
    
    rounds = timeline['rounds']
    detected = timeline['detected']
    
    # Plot detection status
    plt.plot(rounds, detected, linewidth=2, color='#3498db', label='Drift Detected')
    plt.fill_between(rounds, 0, detected, alpha=0.3, color='#3498db')
    
    # Mark injection point
    if timeline['injection_round'] is not None:
        plt.axvline(x=timeline['injection_round'], color='red', linestyle='--',
                   linewidth=2, label=f"Injection (Round {timeline['injection_round']})")
    
    # Mark detection point
    if timeline['detection_round'] is not None:
        plt.axvline(x=timeline['detection_round'], color='green', linestyle='--',
                   linewidth=2, label=f"Detection (Round {timeline['detection_round']})")
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Drift Detected', fontsize=12)
    plt.title('Drift Detection Timeline', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timeline plot to {save_path}")
    
    plt.show()


def generate_latex_table(comparison_df: pd.DataFrame,
                        caption: str = "Comparison of Methods",
                        label: str = "tab:comparison") -> str:
    """
    Generate LaTeX table from comparison DataFrame.
    
    Args:
        comparison_df: Comparison DataFrame
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{l" + "c" * (len(comparison_df.columns) - 1) + "}\n"
    latex += "\\toprule\n"
    
    # Header
    latex += " & ".join(comparison_df.columns) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Data rows
    for _, row in comparison_df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, float):
                if np.isnan(val):
                    row_str.append("-")
                else:
                    row_str.append(f"{val:.3f}")
            elif isinstance(val, bool):
                row_str.append("Yes" if val else "No")
            else:
                row_str.append(str(val))
        latex += " & ".join(row_str) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


# Example usage and testing
if __name__ == "__main__":
    print("Testing Evaluation Metrics...")
    
    # Test 1: ExperimentMetrics
    print("\n1. Testing ExperimentMetrics:")
    metrics = ExperimentMetrics(injection_round=50, affected_clients=[0, 1, 2])
    
    # Simulate rounds
    for round_num in range(100):
        drift = round_num >= 55  # Detected at round 55
        flagged = [0, 1, 3] if drift else []  # Correctly identifies 0, 1 but also flags 3
        
        metrics.update(round_num, drift, flagged, bytes_sent=1024)
    
    final_metrics = metrics.compute_final_metrics(total_rounds=100)
    print(f"  Detection latency: {final_metrics['detection_latency']}")
    print(f"  Client F1: {final_metrics['client_f1']:.3f}")
    print(f"  False alarms: {final_metrics['false_alarms']}")
    
    # Test 2: Method comparison
    print("\n2. Testing method comparison:")
    results = {
        'Federated': {'metrics': {'detection_latency': 5, 'detected': True, 'client_f1': 0.85, 'false_alarm_rate': 0.02}},
        'Centralized': {'metrics': {'detection_latency': 3, 'detected': True, 'client_f1': 0.90, 'false_alarm_rate': 0.01}},
        'Isolated': {'metrics': {'detection_latency': 12, 'detected': True, 'client_f1': 0.60, 'false_alarm_rate': 0.05}}
    }
    
    comparison_df = compare_methods(results)
    print(comparison_df)
    
    # Test 3: LaTeX table generation
    print("\n3. Generating LaTeX table:")
    latex_table = generate_latex_table(comparison_df, caption="Method Comparison")
    print(latex_table[:200] + "...")  # Print first 200 chars
    
    print("\n✅ Evaluation metrics tests passed!")
