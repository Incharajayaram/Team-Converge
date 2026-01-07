"""
Test suite for core modules: privacy, sketching, drift detection, anomaly detection.

Run with: pytest tests/test_core_modules.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.privacy_utils import (
    add_laplace_noise, add_gaussian_noise, 
    PrivacyBudgetTracker, DPHistogram
)
from core.sketch_algorithms import (
    ScoreHistogram, StatisticalSummary, CompressionUtils
)
from core.drift_detection import (
    ks_drift_test, psi_drift_test, js_drift_test,
    EnsembleDriftDetector, DriftAnalyzer
)
from core.anomaly_detection import (
    compute_divergence_matrix, detect_anomalous_clients,
    ClientClusterer, AnomalyScorer
)


class TestPrivacyUtils:
    """Test privacy utilities."""
    
    def test_laplace_noise(self):
        """Test Laplace noise addition."""
        value = 100
        noisy = add_laplace_noise(value, sensitivity=1.0, epsilon=1.0)
        
        # Noise should be added
        assert noisy != value
        # Should be roughly close (with high probability)
        assert abs(noisy - value) < 50  # Very loose bound
    
    def test_gaussian_noise(self):
        """Test Gaussian noise addition."""
        histogram = np.array([10, 20, 15, 8, 5])
        noisy = add_gaussian_noise(histogram, sensitivity=1.0, epsilon=1.0)
        
        assert noisy.shape == histogram.shape
        assert not np.array_equal(noisy, histogram)
    
    def test_privacy_budget_tracker(self):
        """Test privacy budget tracking."""
        tracker = PrivacyBudgetTracker(total_budget=10.0)
        
        # Should allow consumption within budget
        assert tracker.consume(3.0, "query1") == True
        assert tracker.remaining() == 7.0
        
        # Should reject exceeding budget
        assert tracker.consume(8.0, "query2") == False
        assert tracker.remaining() == 7.0
        
        # Should allow within remaining budget
        assert tracker.consume(5.0, "query3") == True
        assert tracker.remaining() == 2.0
    
    def test_dp_histogram(self):
        """Test DP histogram."""
        dp_hist = DPHistogram(num_bins=10, range=(0, 1), epsilon=1.0)
        
        # Add data
        data = np.random.uniform(0, 1, size=100)
        dp_hist.add_values(data)
        
        # Get noisy counts
        noisy = dp_hist.get_noisy_counts(normalize=True)
        
        assert noisy.shape == (10,)
        assert np.isclose(noisy.sum(), 1.0)
        assert np.all(noisy >= 0)


class TestSketchAlgorithms:
    """Test sketch algorithms."""
    
    def test_score_histogram(self):
        """Test ScoreHistogram."""
        hist = ScoreHistogram(num_bins=10, range=(0, 1))
        
        # Add values
        values = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        hist.update_batch(values)
        
        assert hist.total_samples == 5
        assert np.sum(hist.counts) == 5
        
        # Test normalization
        normalized = hist.get_normalized()
        assert np.isclose(normalized.sum(), 1.0)
        
        # Test sparse representation
        sparse = hist.to_sparse_dict()
        assert len(sparse) <= hist.num_bins
    
    def test_statistical_summary(self):
        """Test StatisticalSummary."""
        stats = StatisticalSummary(track_quantiles=True)
        
        # Add data
        data = np.random.normal(0.5, 0.1, size=1000)
        stats.update_batch(data)
        
        # Check statistics
        assert abs(stats.mean - 0.5) < 0.05  # Should be close to true mean
        assert abs(stats.std - 0.1) < 0.02   # Should be close to true std
        assert stats.count == 1000
        
        # Check quantiles
        quantiles = stats.get_quantiles([0.5])
        assert abs(quantiles[0.5] - 0.5) < 0.05
    
    def test_compression(self):
        """Test histogram compression."""
        histogram = np.array([0.3, 0.2, 0.0, 0.0, 0.1, 0.0, 0.4, 0.0])
        
        compressed = CompressionUtils.compress_histogram(histogram, threshold=0.01)
        assert compressed['compression_ratio'] < 1.0
        
        decompressed = CompressionUtils.decompress_histogram(compressed)
        assert decompressed.shape == histogram.shape
        assert np.allclose(histogram, decompressed)


class TestDriftDetection:
    """Test drift detection algorithms."""
    
    @pytest.fixture
    def baseline_hist(self):
        """Create baseline histogram."""
        np.random.seed(42)
        samples = np.random.beta(2, 5, size=1000)
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        return hist / hist.sum()
    
    @pytest.fixture
    def similar_hist(self):
        """Create similar histogram (no drift)."""
        np.random.seed(43)
        samples = np.random.beta(2, 5, size=1000)
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        return hist / hist.sum()
    
    @pytest.fixture
    def drifted_hist(self):
        """Create drifted histogram."""
        np.random.seed(44)
        samples = np.random.beta(5, 2, size=1000)  # Reversed
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        return hist / hist.sum()
    
    def test_ks_test_no_drift(self, baseline_hist, similar_hist):
        """Test KS test with no drift."""
        drift_detected, p_value = ks_drift_test(baseline_hist, similar_hist, threshold=0.01)
        assert drift_detected == False
        assert p_value > 0.01
    
    def test_ks_test_with_drift(self, baseline_hist, drifted_hist):
        """Test KS test with drift."""
        drift_detected, p_value = ks_drift_test(baseline_hist, drifted_hist, threshold=0.01)
        assert drift_detected == True
        assert p_value < 0.01
    
    def test_psi_test(self, baseline_hist, drifted_hist):
        """Test PSI."""
        drift_detected, psi_value = psi_drift_test(baseline_hist, drifted_hist)
        assert drift_detected == True
        assert psi_value > 0.1
    
    def test_js_test(self, baseline_hist, drifted_hist):
        """Test JS divergence."""
        drift_detected, js_div = js_drift_test(baseline_hist, drifted_hist)
        assert drift_detected == True
        assert js_div > 0.1
    
    def test_ensemble_detector(self, baseline_hist, similar_hist, drifted_hist):
        """Test ensemble detector."""
        detector = EnsembleDriftDetector()
        
        # No drift
        drift, scores = detector.detect(baseline_hist, similar_hist)
        assert drift == False
        
        # With drift
        drift, scores = detector.detect(baseline_hist, drifted_hist)
        assert drift == True
        assert scores['num_detections'] >= 2


class TestAnomalyDetection:
    """Test anomaly detection."""
    
    @pytest.fixture
    def client_histograms(self):
        """Create synthetic client histograms."""
        np.random.seed(42)
        
        # Normal clients
        normal = []
        for i in range(8):
            samples = np.random.beta(2, 5, size=1000)
            hist, _ = np.histogram(samples, bins=20, range=(0, 1))
            normal.append(hist / hist.sum())
        
        # Anomalous clients
        anomalous = []
        for i in range(2):
            samples = np.random.beta(5, 2, size=1000)
            hist, _ = np.histogram(samples, bins=20, range=(0, 1))
            anomalous.append(hist / hist.sum())
        
        return normal + anomalous
    
    def test_divergence_matrix(self, client_histograms):
        """Test divergence matrix computation."""
        matrix = compute_divergence_matrix(client_histograms)
        
        assert matrix.shape == (10, 10)
        assert np.allclose(matrix, matrix.T)  # Should be symmetric
        assert np.all(np.diag(matrix) == 0)   # Diagonal should be zero
    
    def test_detect_anomalous_clients(self, client_histograms):
        """Test anomaly detection."""
        anomalous = detect_anomalous_clients(client_histograms, eps=0.15, min_samples=2)
        
        # Should detect the anomalous clients (8, 9)
        assert len(anomalous) > 0
        assert 8 in anomalous or 9 in anomalous
    
    def test_client_clusterer(self, client_histograms):
        """Test ClientClusterer."""
        clusterer = ClientClusterer(eps=0.15, min_samples=2)
        clusterer.fit(client_histograms)
        
        anomalous = clusterer.get_anomalous_clients()
        assert len(anomalous) > 0
        
        summary = clusterer.get_summary()
        assert summary['num_clients'] == 10
        assert summary['num_anomalous'] > 0
    
    def test_anomaly_scorer(self, client_histograms):
        """Test AnomalyScorer."""
        scorer = AnomalyScorer()
        scores = scorer.compute_scores(client_histograms)
        
        assert len(scores) == 10
        
        # Top anomalous should include clients 8, 9
        top = scorer.get_top_anomalous(k=3)
        top_ids = [client_id for client_id, score in top]
        assert 8 in top_ids or 9 in top_ids


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
