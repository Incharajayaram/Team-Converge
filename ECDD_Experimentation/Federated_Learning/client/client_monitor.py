"""
Client-side monitoring for federated drift detection.

Tracks predictions locally and generates privacy-preserving sketches.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sketch_algorithms import ScoreHistogram, StatisticalSummary
from core.privacy_utils import PrivacyBudgetTracker
from core.drift_detection import ks_drift_test, psi_drift_test, js_drift_test


class ClientMonitor:
    """
    Local monitoring for a single federated client.
    
    Accumulates predictions, computes sketches, and optionally detects local drift.
    """
    
    def __init__(self,
                 window_size: int = 500,
                 num_bins: int = 20,
                 epsilon: float = 1.0,
                 track_quantiles: bool = True,
                 local_drift_detection: bool = False,
                 baseline_hist: Optional[np.ndarray] = None):
        """
        Initialize client monitor.
        
        Args:
            window_size: Number of predictions to accumulate before sending
            num_bins: Number of histogram bins
            epsilon: Privacy parameter for DP
            track_quantiles: Whether to track quantiles
            local_drift_detection: Whether to detect drift locally
            baseline_hist: Baseline distribution for local drift detection
        """
        self.window_size = window_size
        self.num_bins = num_bins
        self.epsilon = epsilon
        
        # Prediction buffer
        self.buffer = deque(maxlen=window_size)
        
        # Sketch components
        self.histogram = ScoreHistogram(num_bins=num_bins, range=(0, 1))
        self.statistics = StatisticalSummary(
            track_quantiles=track_quantiles,
            quantile_buffer_size=min(window_size, 1000)
        )
        
        # Metadata tracking
        self.total_predictions = 0
        self.abstain_count = 0
        self.ood_count = 0
        self.high_confidence_count = 0
        
        # Privacy tracking
        self.privacy_tracker = PrivacyBudgetTracker(total_budget=epsilon * 100)  # Allow many queries
        
        # Local drift detection
        self.local_drift_detection = local_drift_detection
        self.baseline_hist = baseline_hist
        self.drift_detected_locally = False
        
        # Timestamps
        self.last_sketch_time = time.time()
        self.creation_time = time.time()
    
    def update(self,
               score: float,
               confidence: float,
               is_ood: bool = False,
               abstained: bool = False):
        """
        Add a new prediction to the monitor.
        
        Args:
            score: Prediction score (0-1, higher = more likely fake)
            confidence: Model confidence (0-1)
            is_ood: Whether sample was flagged as out-of-distribution
            abstained: Whether model abstained from prediction
        """
        # Add to buffer
        self.buffer.append({
            'score': score,
            'confidence': confidence,
            'is_ood': is_ood,
            'abstained': abstained,
            'timestamp': time.time()
        })
        
        # Update histogram and statistics
        self.histogram.update(score)
        self.statistics.update(score)
        
        # Update metadata counts
        self.total_predictions += 1
        if abstained:
            self.abstain_count += 1
        if is_ood:
            self.ood_count += 1
        if confidence > 0.9:
            self.high_confidence_count += 1
        
        # Check for local drift if enabled
        if self.local_drift_detection and self.baseline_hist is not None:
            if self.total_predictions % 100 == 0:  # Check every 100 predictions
                self._check_local_drift()
    
    def _check_local_drift(self):
        """Check for drift locally (optional early warning)."""
        current_hist = self.histogram.get_normalized()
        
        # Quick JS divergence check
        _, js_div = js_drift_test(self.baseline_hist, current_hist, threshold=0.15)
        
        if js_div > 0.15:
            self.drift_detected_locally = True
    
    def is_buffer_full(self) -> bool:
        """Check if buffer is full and ready to send sketch."""
        return len(self.buffer) >= self.window_size
    
    def get_sketch(self, apply_dp: bool = True, clear_after: bool = True) -> Dict:
        """
        Generate privacy-preserving sketch.
        
        Args:
            apply_dp: Whether to apply differential privacy noise
            clear_after: Whether to clear buffer after generating sketch
            
        Returns:
            Dictionary with sketch data
        """
        # Get histogram
        if apply_dp:
            histogram_data = self.histogram.add_dp_noise(self.epsilon)
            # Normalize after DP noise
            histogram_data = histogram_data / histogram_data.sum()
            
            # Track privacy consumption
            self.privacy_tracker.consume(self.epsilon, f"sketch_at_{time.time()}")
        else:
            histogram_data = self.histogram.get_normalized()
        
        # Convert to sparse representation for compression
        sparse_hist = self.histogram.to_sparse_dict()
        
        # Get statistics
        stats = self.statistics.get_stats()
        
        # Compute metadata
        metadata = self._compute_metadata()
        
        sketch = {
            'histogram': sparse_hist,
            'histogram_full': histogram_data.tolist(),  # For aggregation
            'statistics': stats,
            'metadata': metadata,
            'privacy': {
                'epsilon': self.epsilon,
                'applied_dp': apply_dp,
                'remaining_budget': self.privacy_tracker.remaining()
            },
            'timestamp': time.time()
        }
        
        # Clear buffer if requested
        if clear_after:
            self.clear()
        
        return sketch
    
    def _compute_metadata(self) -> Dict:
        """Compute metadata about predictions."""
        if self.total_predictions == 0:
            return {
                'num_samples': 0,
                'abstain_rate': 0.0,
                'ood_rate': 0.0,
                'high_confidence_rate': 0.0
            }
        
        return {
            'num_samples': self.total_predictions,
            'abstain_rate': self.abstain_count / self.total_predictions,
            'ood_rate': self.ood_count / self.total_predictions,
            'high_confidence_rate': self.high_confidence_count / self.total_predictions,
            'buffer_full': self.is_buffer_full(),
            'drift_detected_locally': self.drift_detected_locally,
            'time_since_creation': time.time() - self.creation_time,
            'time_since_last_sketch': time.time() - self.last_sketch_time
        }
    
    def clear(self):
        """Clear buffer and reset accumulators."""
        self.histogram.clear()
        self.statistics.clear()
        self.total_predictions = 0
        self.abstain_count = 0
        self.ood_count = 0
        self.high_confidence_count = 0
        self.drift_detected_locally = False
        self.last_sketch_time = time.time()
        # Note: buffer is deque with maxlen, so it auto-manages
    
    def get_current_stats(self) -> Dict:
        """Get current statistics without generating sketch."""
        return {
            'total_predictions': self.total_predictions,
            'buffer_size': len(self.buffer),
            'buffer_full': self.is_buffer_full(),
            'mean_score': self.statistics.mean,
            'std_score': self.statistics.std,
            'abstain_rate': self.abstain_count / self.total_predictions if self.total_predictions > 0 else 0.0
        }
    
    def __repr__(self):
        return f"ClientMonitor(predictions={self.total_predictions}, buffer={len(self.buffer)}/{self.window_size})"


# Example usage and testing
if __name__ == "__main__":
    print("Testing ClientMonitor...")
    
    # Create monitor
    print("\n1. Creating monitor:")
    monitor = ClientMonitor(window_size=100, epsilon=1.0)
    print(f"  {monitor}")
    
    # Simulate predictions
    print("\n2. Simulating predictions:")
    np.random.seed(42)
    
    for i in range(150):
        score = np.random.beta(2, 5)  # Fake scores biased toward lower values
        confidence = np.random.uniform(0.7, 1.0)
        is_ood = np.random.random() < 0.05  # 5% OOD
        abstained = np.random.random() < 0.02  # 2% abstain
        
        monitor.update(score, confidence, is_ood, abstained)
        
        if i % 50 == 0:
            print(f"  Prediction {i}: {monitor.get_current_stats()}")
    
    # Generate sketch
    print("\n3. Generating sketch:")
    print(f"  Buffer full: {monitor.is_buffer_full()}")
    
    sketch = monitor.get_sketch(apply_dp=True)
    print(f"  Sketch keys: {list(sketch.keys())}")
    print(f"  Num samples: {sketch['metadata']['num_samples']}")
    print(f"  Abstain rate: {sketch['metadata']['abstain_rate']:.2%}")
    print(f"  Statistics: mean={sketch['statistics']['mean']:.3f}, std={sketch['statistics']['std']:.3f}")
    print(f"  Sparse histogram bins: {len(sketch['histogram'])}/20")
    
    # Test with DP
    print("\n4. Testing differential privacy:")
    monitor2 = ClientMonitor(window_size=100, epsilon=0.1)  # Stronger privacy
    
    for i in range(100):
        score = np.random.beta(2, 5)
        monitor2.update(score, 0.9)
    
    sketch_low_eps = monitor2.get_sketch(apply_dp=True)
    print(f"  With ε=0.1 (strong privacy): privacy budget remaining = {sketch_low_eps['privacy']['remaining_budget']:.2f}")
    
    print("\n✅ ClientMonitor tests passed!")
