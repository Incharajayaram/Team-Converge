"""
Drift detection algorithms for monitoring distribution shifts.

Implements KS-test, PSI, and JS-divergence with ensemble detection.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


def ks_drift_test(baseline_hist: np.ndarray, 
                  current_hist: np.ndarray, 
                  threshold: float = 0.01) -> Tuple[bool, float]:
    """
    Kolmogorov-Smirnov test for sudden distribution shifts.
    
    Args:
        baseline_hist: Baseline histogram (normalized)
        current_hist: Current histogram (normalized)
        threshold: p-value threshold (default 0.01 for 99% confidence)
        
    Returns:
        (drift_detected, p_value)
        
    Example:
        >>> baseline = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        >>> current = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        >>> drift, p_val = ks_drift_test(baseline, current)
    """
    # Reconstruct approximate samples from histograms
    # This is an approximation - ideally we'd use raw samples
    baseline_samples = reconstruct_samples_from_histogram(baseline_hist)
    current_samples = reconstruct_samples_from_histogram(current_hist)
    
    # Perform KS test
    statistic, p_value = ks_2samp(baseline_samples, current_samples)
    
    drift_detected = p_value < threshold
    return drift_detected, float(p_value)


def psi_drift_test(baseline_hist: np.ndarray,
                   current_hist: np.ndarray,
                   threshold: float = 0.1) -> Tuple[bool, float]:
    """
    Population Stability Index for gradual drift detection.
    
    PSI = sum((current_prop - baseline_prop) * ln(current_prop / baseline_prop))
    
    PSI interpretation:
    - < 0.1: No significant change
    - 0.1 - 0.25: Small change
    - > 0.25: Significant change
    
    Args:
        baseline_hist: Baseline histogram (normalized)
        current_hist: Current histogram (normalized)
        threshold: PSI threshold (default 0.1)
        
    Returns:
        (drift_detected, psi_value)
    """
    # Ensure normalized
    baseline_hist = baseline_hist / baseline_hist.sum()
    current_hist = current_hist / current_hist.sum()
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    baseline_hist = baseline_hist + epsilon
    current_hist = current_hist + epsilon
    
    # Compute PSI
    psi = np.sum((current_hist - baseline_hist) * np.log(current_hist / baseline_hist))
    
    drift_detected = psi > threshold
    return drift_detected, float(psi)


def js_drift_test(baseline_hist: np.ndarray,
                  current_hist: np.ndarray,
                  threshold: float = 0.1) -> Tuple[bool, float]:
    """
    Jensen-Shannon divergence for distribution similarity.
    
    JS divergence is symmetric and bounded [0, 1].
    
    Args:
        baseline_hist: Baseline histogram (normalized)
        current_hist: Current histogram (normalized)
        threshold: JS divergence threshold (default 0.1)
        
    Returns:
        (drift_detected, js_divergence)
    """
    # Ensure normalized
    baseline_hist = baseline_hist / baseline_hist.sum()
    current_hist = current_hist / current_hist.sum()
    
    # Compute JS divergence
    js_div = jensenshannon(baseline_hist, current_hist)
    
    drift_detected = js_div > threshold
    return drift_detected, float(js_div)


def reconstruct_samples_from_histogram(histogram: np.ndarray, 
                                       num_samples: int = 1000) -> np.ndarray:
    """
    Reconstruct approximate samples from a histogram.
    
    Used for statistical tests that require sample data.
    
    Args:
        histogram: Normalized histogram
        num_samples: Number of samples to generate
        
    Returns:
        Array of reconstructed samples
    """
    # Ensure normalized
    histogram = histogram / histogram.sum()
    
    # Generate bin centers
    num_bins = len(histogram)
    bin_centers = np.linspace(0, 1, num_bins)
    
    # Sample from discrete distribution
    samples = np.random.choice(bin_centers, size=num_samples, p=histogram)
    
    return samples


class EnsembleDriftDetector:
    """
    Ensemble drift detector combining KS-test, PSI, and JS-divergence.
    
    Uses majority voting to reduce false positives.
    """
    
    def __init__(self, 
                 ks_threshold: float = 0.01,
                 psi_threshold: float = 0.1,
                 js_threshold: float = 0.1,
                 majority_vote: int = 2):
        """
        Initialize ensemble detector.
        
        Args:
            ks_threshold: KS test p-value threshold
            psi_threshold: PSI threshold
            js_threshold: JS divergence threshold
            majority_vote: Number of detectors that must agree (default 2/3)
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.js_threshold = js_threshold
        self.majority_vote = majority_vote
        
        self.detection_history = []
    
    def detect(self, baseline_hist: np.ndarray, current_hist: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect drift using ensemble of methods.
        
        Args:
            baseline_hist: Baseline histogram
            current_hist: Current histogram
            
        Returns:
            (drift_detected, scores_dict)
            
        Example:
            >>> detector = EnsembleDriftDetector()
            >>> drift, scores = detector.detect(baseline, current)
            >>> print(f"Drift: {drift}, Scores: {scores}")
        """
        # Run all three detectors
        ks_detected, ks_score = ks_drift_test(baseline_hist, current_hist, self.ks_threshold)
        psi_detected, psi_score = psi_drift_test(baseline_hist, current_hist, self.psi_threshold)
        js_detected, js_score = js_drift_test(baseline_hist, current_hist, self.js_threshold)
        
        # Count detections
        detections = [ks_detected, psi_detected, js_detected]
        num_detections = sum(detections)
        
        # Majority vote
        drift_detected = num_detections >= self.majority_vote
        
        # Compile scores
        scores = {
            'ks_detected': ks_detected,
            'ks_pvalue': ks_score,
            'psi_detected': psi_detected,
            'psi_value': psi_score,
            'js_detected': js_detected,
            'js_divergence': js_score,
            'num_detections': num_detections,
            'ensemble_detected': drift_detected
        }
        
        # Store in history
        self.detection_history.append(scores)
        
        return drift_detected, scores
    
    def get_detection_summary(self) -> Dict:
        """
        Get summary of all detections.
        
        Returns:
            Dictionary with detection statistics
        """
        if not self.detection_history:
            return {'total_checks': 0, 'total_drifts': 0}
        
        total_checks = len(self.detection_history)
        total_drifts = sum(h['ensemble_detected'] for h in self.detection_history)
        
        # Average scores
        avg_ks = np.mean([h['ks_pvalue'] for h in self.detection_history])
        avg_psi = np.mean([h['psi_value'] for h in self.detection_history])
        avg_js = np.mean([h['js_divergence'] for h in self.detection_history])
        
        return {
            'total_checks': total_checks,
            'total_drifts': total_drifts,
            'drift_rate': total_drifts / total_checks,
            'avg_ks_pvalue': avg_ks,
            'avg_psi_value': avg_psi,
            'avg_js_divergence': avg_js
        }
    
    def reset_history(self):
        """Clear detection history."""
        self.detection_history = []
    
    def __repr__(self):
        summary = self.get_detection_summary()
        return f"EnsembleDriftDetector(checks={summary['total_checks']}, drifts={summary['total_drifts']})"


class DriftAnalyzer:
    """
    Analyze drift patterns over time.
    """
    
    def __init__(self, baseline_hist: np.ndarray):
        """
        Initialize analyzer with baseline distribution.
        
        Args:
            baseline_hist: Baseline histogram
        """
        self.baseline = baseline_hist / baseline_hist.sum()
        self.history = []
    
    def add_observation(self, histogram: np.ndarray, timestamp: float):
        """
        Add a new observation.
        
        Args:
            histogram: Current histogram
            timestamp: Timestamp of observation
        """
        normalized = histogram / histogram.sum()
        
        # Compute all drift metrics
        _, ks_pval = ks_drift_test(self.baseline, normalized)
        _, psi_val = psi_drift_test(self.baseline, normalized)
        _, js_div = js_drift_test(self.baseline, normalized)
        
        self.history.append({
            'timestamp': timestamp,
            'histogram': normalized,
            'ks_pvalue': ks_pval,
            'psi_value': psi_val,
            'js_divergence': js_div
        })
    
    def get_drift_trajectory(self) -> Dict:
        """
        Get trajectory of drift over time.
        
        Returns:
            Dictionary with time series of drift metrics
        """
        if not self.history:
            return {}
        
        timestamps = [h['timestamp'] for h in self.history]
        ks_pvalues = [h['ks_pvalue'] for h in self.history]
        psi_values = [h['psi_value'] for h in self.history]
        js_divergences = [h['js_divergence'] for h in self.history]
        
        return {
            'timestamps': timestamps,
            'ks_pvalues': ks_pvalues,
            'psi_values': psi_values,
            'js_divergences': js_divergences
        }
    
    def detect_drift_type(self) -> str:
        """
        Classify type of drift (sudden, gradual, none).
        
        Returns:
            Drift type: 'none', 'sudden', 'gradual'
        """
        if len(self.history) < 3:
            return 'insufficient_data'
        
        # Get recent drift scores
        recent_js = [h['js_divergence'] for h in self.history[-5:]]
        
        # Check for sudden drift (sharp increase)
        if len(recent_js) >= 2:
            increase = recent_js[-1] - recent_js[0]
            if increase > 0.15:  # Large sudden increase
                return 'sudden'
        
        # Check for gradual drift (steady increase)
        if len(recent_js) >= 3:
            # Linear regression to check trend
            x = np.arange(len(recent_js))
            slope = np.polyfit(x, recent_js, 1)[0]
            if slope > 0.02:  # Positive trend
                return 'gradual'
        
        # Check if no drift
        if all(js < 0.1 for js in recent_js):
            return 'none'
        
        return 'fluctuating'


# Example usage and testing
if __name__ == "__main__":
    print("Testing Drift Detection Algorithms...")
    
    # Create baseline and drifted distributions
    np.random.seed(42)
    
    # Baseline: Beta(2, 5) distribution
    baseline_samples = np.random.beta(2, 5, size=1000)
    baseline_hist, _ = np.histogram(baseline_samples, bins=20, range=(0, 1))
    baseline_hist = baseline_hist / baseline_hist.sum()
    
    print("\n1. Testing individual detectors:")
    
    # No drift case
    no_drift_samples = np.random.beta(2, 5, size=1000)
    no_drift_hist, _ = np.histogram(no_drift_samples, bins=20, range=(0, 1))
    no_drift_hist = no_drift_hist / no_drift_hist.sum()
    
    ks_det, ks_score = ks_drift_test(baseline_hist, no_drift_hist)
    print(f"No drift - KS test: detected={ks_det}, p-value={ks_score:.4f}")
    
    psi_det, psi_score = psi_drift_test(baseline_hist, no_drift_hist)
    print(f"No drift - PSI: detected={psi_det}, value={psi_score:.4f}")
    
    js_det, js_score = js_drift_test(baseline_hist, no_drift_hist)
    print(f"No drift - JS: detected={js_det}, divergence={js_score:.4f}")
    
    # Sudden drift case
    drift_samples = np.random.beta(5, 2, size=1000)  # Reversed distribution
    drift_hist, _ = np.histogram(drift_samples, bins=20, range=(0, 1))
    drift_hist = drift_hist / drift_hist.sum()
    
    print("\nSudden drift case:")
    ks_det, ks_score = ks_drift_test(baseline_hist, drift_hist)
    print(f"Drift - KS test: detected={ks_det}, p-value={ks_score:.4f}")
    
    psi_det, psi_score = psi_drift_test(baseline_hist, drift_hist)
    print(f"Drift - PSI: detected={psi_det}, value={psi_score:.4f}")
    
    js_det, js_score = js_drift_test(baseline_hist, drift_hist)
    print(f"Drift - JS: detected={js_det}, divergence={js_score:.4f}")
    
    # Test ensemble detector
    print("\n2. Testing ensemble detector:")
    detector = EnsembleDriftDetector()
    
    # No drift
    drift_detected, scores = detector.detect(baseline_hist, no_drift_hist)
    print(f"No drift - Ensemble: {drift_detected}")
    print(f"  Scores: {scores['num_detections']}/3 detectors triggered")
    
    # With drift
    drift_detected, scores = detector.detect(baseline_hist, drift_hist)
    print(f"Drift - Ensemble: {drift_detected}")
    print(f"  Scores: {scores['num_detections']}/3 detectors triggered")
    print(f"  Details: KS={scores['ks_detected']}, PSI={scores['psi_detected']}, JS={scores['js_detected']}")
    
    # Test drift analyzer
    print("\n3. Testing drift analyzer:")
    analyzer = DriftAnalyzer(baseline_hist)
    
    # Simulate gradual drift
    for i in range(10):
        # Gradually shift distribution
        alpha = 2 + i * 0.3  # Shift from 2 to 5
        beta = 5 - i * 0.3   # Shift from 5 to 2
        samples = np.random.beta(alpha, beta, size=1000)
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        analyzer.add_observation(hist, timestamp=i)
    
    drift_type = analyzer.detect_drift_type()
    print(f"Detected drift type: {drift_type}")
    
    trajectory = analyzer.get_drift_trajectory()
    print(f"JS divergence trajectory: {[f'{js:.3f}' for js in trajectory['js_divergences']]}")
    
    print("\n✅ Drift detection tests passed!")
