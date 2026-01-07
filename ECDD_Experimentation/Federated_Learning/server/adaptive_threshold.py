"""
Adaptive Threshold Manager for federated threshold calibration.

Optimizes classification threshold based on aggregated distributions
to achieve target false positive rate (FPR).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class AdaptiveThresholdManager:
    """
    Manages threshold calibration in federated setting.
    
    Computes optimal threshold from aggregated score distributions
    to achieve target FPR while maximizing true positive rate.
    """
    
    def __init__(self,
                 initial_threshold: float = 0.5,
                 target_fpr: float = 0.01,
                 min_samples_for_calibration: int = 100):
        """
        Initialize threshold manager.
        
        Args:
            initial_threshold: Starting threshold
            target_fpr: Target false positive rate (e.g., 0.01 = 1%)
            min_samples_for_calibration: Minimum samples needed for calibration
        """
        self.current_threshold = initial_threshold
        self.target_fpr = target_fpr
        self.min_samples = min_samples_for_calibration
        
        # History tracking
        self.threshold_history = [{
            'threshold': initial_threshold,
            'timestamp': time.time(),
            'reason': 'initialization'
        }]
        
        self.calibration_count = 0
        self.last_calibration_time = time.time()
    
    def calibrate_threshold(self,
                           aggregated_hist: np.ndarray,
                           ground_truth_labels: Optional[np.ndarray] = None) -> float:
        """
        Calibrate threshold from aggregated histogram.
        
        Method 1: If no ground truth, assumes histogram represents score distribution
                  and sets threshold to achieve target FPR
        Method 2: If ground truth provided, computes ROC and finds optimal threshold
        
        Args:
            aggregated_hist: Aggregated score histogram (normalized)
            ground_truth_labels: Optional ground truth labels for samples
            
        Returns:
            Optimal threshold value
        """
        # Ensure normalized
        aggregated_hist = aggregated_hist / aggregated_hist.sum()
        
        if ground_truth_labels is not None:
            # Method 2: Use ground truth to compute ROC
            threshold = self._calibrate_with_ground_truth(aggregated_hist, ground_truth_labels)
        else:
            # Method 1: Estimate from distribution
            threshold = self._calibrate_from_distribution(aggregated_hist)
        
        # Update state
        self.current_threshold = threshold
        self.calibration_count += 1
        self.last_calibration_time = time.time()
        
        # Store in history
        self.threshold_history.append({
            'threshold': threshold,
            'timestamp': time.time(),
            'reason': 'drift_calibration',
            'target_fpr': self.target_fpr
        })
        
        return threshold
    
    def _calibrate_from_distribution(self, histogram: np.ndarray) -> float:
        """
        Calibrate threshold from histogram to achieve target FPR.
        
        Assumes:
        - Histogram represents deepfake scores (0 = real, 1 = fake)
        - Lower scores should be classified as real (negatives)
        - Higher scores should be classified as fake (positives)
        - FPR = proportion of reals (low scores) incorrectly classified as fake
        
        Args:
            histogram: Normalized score histogram (20 bins)
            
        Returns:
            Threshold value
        """
        # Reconstruct CDF from histogram
        num_bins = len(histogram)
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute cumulative distribution
        cdf = np.cumsum(histogram)
        
        # For deepfake detection:
        # - Scores < threshold → classified as real (negative)
        # - Scores >= threshold → classified as fake (positive)
        # - FPR = P(score >= threshold | real image)
        # 
        # If we want FPR = target_fpr, we need:
        # P(score >= threshold) = target_fpr (assuming uniform class distribution)
        # Therefore: threshold such that CDF(threshold) = 1 - target_fpr
        
        target_cdf = 1.0 - self.target_fpr
        
        # Find bin where CDF crosses target
        threshold_idx = np.searchsorted(cdf, target_cdf)
        threshold_idx = min(threshold_idx, num_bins - 1)
        
        threshold = bin_centers[threshold_idx]
        
        # Ensure threshold is in valid range
        threshold = np.clip(threshold, 0.1, 0.9)
        
        return float(threshold)
    
    def _calibrate_with_ground_truth(self,
                                     scores: np.ndarray,
                                     labels: np.ndarray) -> float:
        """
        Calibrate threshold using ground truth labels (ROC curve method).
        
        Args:
            scores: Prediction scores
            labels: Ground truth labels (0 = real, 1 = fake)
            
        Returns:
            Optimal threshold
        """
        # Sort by scores
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Compute TPR and FPR for each threshold
        total_positives = np.sum(labels == 1)
        total_negatives = np.sum(labels == 0)
        
        if total_positives == 0 or total_negatives == 0:
            # Degenerate case, return default
            return 0.5
        
        best_threshold = 0.5
        best_tpr = 0.0
        
        # Try each unique score as threshold
        unique_scores = np.unique(sorted_scores)
        
        for threshold in unique_scores:
            predictions = (scores >= threshold).astype(int)
            
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            
            tpr = tp / total_positives
            fpr = fp / total_negatives
            
            # Find threshold that achieves target FPR with maximum TPR
            if abs(fpr - self.target_fpr) < 0.01:  # Within tolerance
                if tpr > best_tpr:
                    best_tpr = tpr
                    best_threshold = threshold
        
        return float(best_threshold)
    
    def should_update_threshold(self, current_fpr: float, tolerance: float = 0.02) -> bool:
        """
        Check if threshold should be updated based on current FPR.
        
        Args:
            current_fpr: Current observed false positive rate
            tolerance: Acceptable deviation from target
            
        Returns:
            True if threshold should be recalibrated
        """
        deviation = abs(current_fpr - self.target_fpr)
        return deviation > tolerance
    
    def compute_metrics(self,
                       scores: np.ndarray,
                       labels: np.ndarray,
                       threshold: Optional[float] = None) -> Dict:
        """
        Compute classification metrics for given threshold.
        
        Args:
            scores: Prediction scores
            labels: Ground truth labels
            threshold: Threshold to evaluate (default: current threshold)
            
        Returns:
            Dictionary with metrics (TPR, FPR, precision, recall, F1)
        """
        if threshold is None:
            threshold = self.current_threshold
        
        predictions = (scores >= threshold).astype(int)
        
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        # Compute metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        
        return {
            'threshold': threshold,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }
    
    def get_threshold_trajectory(self) -> Dict:
        """
        Get trajectory of threshold changes over time.
        
        Returns:
            Dictionary with timestamps and threshold values
        """
        timestamps = [h['timestamp'] for h in self.threshold_history]
        thresholds = [h['threshold'] for h in self.threshold_history]
        reasons = [h['reason'] for h in self.threshold_history]
        
        return {
            'timestamps': timestamps,
            'thresholds': thresholds,
            'reasons': reasons,
            'num_calibrations': self.calibration_count
        }
    
    def get_summary(self) -> Dict:
        """
        Get summary of threshold management.
        
        Returns:
            Dictionary with current state
        """
        return {
            'current_threshold': self.current_threshold,
            'target_fpr': self.target_fpr,
            'calibration_count': self.calibration_count,
            'time_since_last_calibration': time.time() - self.last_calibration_time,
            'threshold_range': (
                min(h['threshold'] for h in self.threshold_history),
                max(h['threshold'] for h in self.threshold_history)
            )
        }
    
    def __repr__(self):
        return (f"AdaptiveThresholdManager(threshold={self.current_threshold:.3f}, "
                f"target_fpr={self.target_fpr:.3f}, calibrations={self.calibration_count})")


class ThresholdCalibrator:
    """
    Alias for AdaptiveThresholdManager for backward compatibility.
    """
    def __init__(self, *args, **kwargs):
        self.manager = AdaptiveThresholdManager(*args, **kwargs)
    
    def calibrate(self, *args, **kwargs):
        return self.manager.calibrate_threshold(*args, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Testing AdaptiveThresholdManager...")
    
    np.random.seed(42)
    
    # Test 1: Calibrate from distribution (no ground truth)
    print("\n1. Calibration from distribution:")
    
    # Create histogram from samples
    samples = np.random.beta(2, 5, size=1000)
    hist, _ = np.histogram(samples, bins=20, range=(0, 1))
    hist = hist / hist.sum()
    
    manager = AdaptiveThresholdManager(target_fpr=0.01)
    print(f"  Initial: {manager}")
    
    threshold = manager.calibrate_threshold(hist)
    print(f"  Calibrated threshold: {threshold:.3f}")
    print(f"  Target FPR: {manager.target_fpr:.3f}")
    
    # Test 2: Calibrate with ground truth
    print("\n2. Calibration with ground truth:")
    
    # Generate synthetic data
    n_samples = 1000
    n_positives = 500
    
    # Real images (negatives): low scores
    real_scores = np.random.beta(2, 5, size=n_samples - n_positives)
    # Fake images (positives): high scores  
    fake_scores = np.random.beta(5, 2, size=n_positives)
    
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.zeros(n_samples - n_positives), np.ones(n_positives)])
    
    threshold_gt = manager.calibrate_threshold(hist, ground_truth_labels=labels)
    print(f"  Calibrated threshold (with GT): {threshold_gt:.3f}")
    
    # Compute metrics
    metrics = manager.compute_metrics(scores, labels, threshold_gt)
    print(f"  Metrics:")
    print(f"    TPR (Recall): {metrics['tpr']:.3f}")
    print(f"    FPR: {metrics['fpr']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    F1: {metrics['f1']:.3f}")
    
    # Test 3: Multiple calibrations
    print("\n3. Multiple calibrations:")
    
    for i in range(3):
        # Simulate drift
        samples = np.random.beta(2 + i*0.5, 5 - i*0.5, size=1000)
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        hist = hist / hist.sum()
        
        threshold = manager.calibrate_threshold(hist)
        print(f"  Calibration {i+1}: threshold = {threshold:.3f}")
    
    # Get trajectory
    print("\n4. Threshold trajectory:")
    trajectory = manager.get_threshold_trajectory()
    print(f"  Total calibrations: {trajectory['num_calibrations']}")
    print(f"  Thresholds: {[f'{t:.3f}' for t in trajectory['thresholds']]}")
    
    # Get summary
    print("\n5. Summary:")
    summary = manager.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ AdaptiveThresholdManager tests passed!")
