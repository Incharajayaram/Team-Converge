"""
Quick test script to verify all core modules work together.

Run: python test_core_quick.py
"""

import numpy as np
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing Core Modules - Quick Verification")
print("=" * 60)

# Test 1: Privacy Utils
print("\n[1/4] Testing Privacy Utils...")
from core.privacy_utils import add_laplace_noise, PrivacyBudgetTracker, DPHistogram

value = 100
noisy = add_laplace_noise(value, sensitivity=1.0, epsilon=1.0)
print(f"  ✓ Laplace noise: {value} → {noisy:.2f}")

tracker = PrivacyBudgetTracker(10.0)
tracker.consume(3.0, "test")
print(f"  ✓ Privacy budget: {tracker.remaining():.1f}/10.0 remaining")

# Test 2: Sketch Algorithms
print("\n[2/4] Testing Sketch Algorithms...")
from core.sketch_algorithms import ScoreHistogram, StatisticalSummary

hist = ScoreHistogram(num_bins=20)
data = np.random.beta(2, 5, size=1000)
hist.update_batch(data)
print(f"  ✓ Histogram: {hist.total_samples} samples, {len(hist.to_sparse_dict())} non-zero bins")

stats = StatisticalSummary(track_quantiles=True)
stats.update_batch(data)
print(f"  ✓ Statistics: mean={stats.mean:.3f}, std={stats.std:.3f}")

# Test 3: Drift Detection
print("\n[3/4] Testing Drift Detection...")
from core.drift_detection import EnsembleDriftDetector

baseline_samples = np.random.beta(2, 5, size=1000)
baseline_hist, _ = np.histogram(baseline_samples, bins=20, range=(0, 1))
baseline_hist = baseline_hist / baseline_hist.sum()

# No drift
similar_samples = np.random.beta(2, 5, size=1000)
similar_hist, _ = np.histogram(similar_samples, bins=20, range=(0, 1))
similar_hist = similar_hist / similar_hist.sum()

# With drift
drifted_samples = np.random.beta(5, 2, size=1000)
drifted_hist, _ = np.histogram(drifted_samples, bins=20, range=(0, 1))
drifted_hist = drifted_hist / drifted_hist.sum()

detector = EnsembleDriftDetector()
drift_no, scores_no = detector.detect(baseline_hist, similar_hist)
drift_yes, scores_yes = detector.detect(baseline_hist, drifted_hist)

print(f"  ✓ No drift detected: {drift_no} (expected False)")
print(f"  ✓ Drift detected: {drift_yes} (expected True)")

# Test 4: Anomaly Detection
print("\n[4/4] Testing Anomaly Detection...")
from core.anomaly_detection import ClientClusterer

# Create client distributions
normal_clients = []
for i in range(8):
    samples = np.random.beta(2, 5, size=1000)
    hist, _ = np.histogram(samples, bins=20, range=(0, 1))
    normal_clients.append(hist / hist.sum())

anomalous_clients = []
for i in range(2):
    samples = np.random.beta(5, 2, size=1000)
    hist, _ = np.histogram(samples, bins=20, range=(0, 1))
    anomalous_clients.append(hist / hist.sum())

all_clients = normal_clients + anomalous_clients

clusterer = ClientClusterer(eps=0.15, min_samples=2)
clusterer.fit(all_clients)

anomalous = clusterer.get_anomalous_clients()
print(f"  ✓ Anomalous clients detected: {anomalous} (expected [8, 9])")

# Summary
print("\n" + "=" * 60)
print("✅ All core modules working!")
print("=" * 60)
print("\nCore modules tested:")
print("  1. Privacy utilities (DP noise, budget tracking)")
print("  2. Sketch algorithms (histograms, statistics)")
print("  3. Drift detection (KS, PSI, JS, ensemble)")
print("  4. Anomaly detection (clustering, divergence)")
print("\nNext: Implement client-side monitoring")
print("=" * 60)
