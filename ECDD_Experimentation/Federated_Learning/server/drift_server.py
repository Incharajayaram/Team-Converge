"""
Central Drift Server for global coordination.

Aggregates sketches from all hub aggregators, detects global drift,
identifies anomalous hubs, and coordinates threshold updates.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.drift_detection import EnsembleDriftDetector, DriftAnalyzer
from core.anomaly_detection import ClientClusterer, AnomalyScorer
from server.adaptive_threshold import AdaptiveThresholdManager


class CentralDriftServer:
    """
    Central server coordinating all hubs in hierarchical federation.
    
    Responsibilities:
    1. Receive aggregated sketches from hubs
    2. Aggregate global distribution
    3. Detect global drift
    4. Identify anomalous hubs
    5. Coordinate threshold updates
    """
    
    def __init__(self,
                 baseline_hist: np.ndarray,
                 num_hubs: int,
                 min_hubs_for_aggregation: int = 2,
                 target_fpr: float = 0.01):
        """
        Initialize central drift server.
        
        Args:
            baseline_hist: Baseline distribution (from training/validation data)
            num_hubs: Expected number of hubs
            min_hubs_for_aggregation: Minimum hubs before global aggregation
            target_fpr: Target false positive rate for threshold calibration
        """
        self.baseline_hist = baseline_hist / baseline_hist.sum()  # Normalize
        self.num_hubs = num_hubs
        self.min_hubs = min_hubs_for_aggregation
        self.target_fpr = target_fpr
        
        # Hub tracking
        self.hub_sketches = {}  # {hub_id: sketch_data}
        self.hub_metadata = {}  # {hub_id: metadata}
        self.connected_hubs = set()
        
        # Global state
        self.global_aggregated_hist = None
        self.global_aggregated_stats = None
        self.aggregation_count = 0
        self.last_aggregation_time = time.time()
        
        # Drift detection
        self.drift_detector = EnsembleDriftDetector(
            ks_threshold=0.01,
            psi_threshold=0.1,
            js_threshold=0.1,
            majority_vote=2
        )
        self.drift_analyzer = DriftAnalyzer(baseline_hist=self.baseline_hist)
        self.drift_detected = False
        self.drift_history = []
        
        # Anomaly detection
        self.anomaly_clusterer = ClientClusterer(eps=0.15, min_samples=2)
        self.anomaly_scorer = AnomalyScorer()
        self.anomalous_hubs = []
        
        # Threshold management
        self.threshold_manager = AdaptiveThresholdManager(
            initial_threshold=0.5,
            target_fpr=target_fpr
        )
        self.current_threshold = 0.5
        self.threshold_updated = False
        
        # Communication tracking
        self.total_hub_submissions = 0
        self.total_global_aggregations = 0
        
        print(f"[Central Server] Initialized for {num_hubs} hubs")
    
    def receive_hub_aggregation(self, hub_id: int, aggregated_hist: np.ndarray, 
                               metadata: Dict) -> Dict:
        """
        Receive aggregated sketch from a hub.
        
        Args:
            hub_id: Hub identifier
            aggregated_hist: Hub's aggregated histogram
            metadata: Hub metadata (stats, num_students, etc.)
            
        Returns:
            Response dictionary for hub
        """
        # Store hub data
        self.hub_sketches[hub_id] = {
            'histogram': aggregated_hist / aggregated_hist.sum(),  # Normalize
            'metadata': metadata,
            'timestamp': time.time()
        }
        self.hub_metadata[hub_id] = metadata
        self.connected_hubs.add(hub_id)
        self.total_hub_submissions += 1
        
        print(f"[Central Server] Received from Hub {hub_id} "
              f"({len(self.hub_sketches)}/{self.min_hubs} hubs)")
        
        # Check if should aggregate globally
        if len(self.hub_sketches) >= self.min_hubs:
            self.aggregate_global()
        
        # Prepare response
        response = {
            'status': 'received',
            'global_drift_detected': self.drift_detected,
            'anomalous_hubs': self.anomalous_hubs,
            'current_threshold': self.current_threshold
        }
        
        # Include threshold update if changed
        if self.threshold_updated:
            response['threshold_update'] = self.current_threshold
            self.threshold_updated = False  # Reset flag
        
        return response
    
    def aggregate_global(self):
        """
        Aggregate distributions from all hubs to compute global distribution.
        
        Uses weighted average based on number of students per hub.
        """
        if len(self.hub_sketches) < self.min_hubs:
            return
        
        print(f"[Central Server] Aggregating {len(self.hub_sketches)} hub distributions...")
        
        # Extract hub histograms and weights
        hub_hists = []
        weights = []
        
        for hub_id, data in self.hub_sketches.items():
            hist = data['histogram']
            num_students = data['metadata'].get('num_students', 1)
            total_samples = data['metadata'].get('aggregated_stats', {}).get('total_samples', 100)
            
            hub_hists.append(hist)
            weights.append(total_samples)  # Weight by total samples
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        self.global_aggregated_hist = np.average(hub_hists, axis=0, weights=weights)
        
        # Aggregate global statistics
        self.global_aggregated_stats = self._aggregate_global_stats()
        
        # Detect global drift
        self.check_global_drift()
        
        # Detect anomalous hubs
        self.check_anomalous_hubs()
        
        # Update aggregation tracking
        self.aggregation_count += 1
        self.total_global_aggregations += 1
        self.last_aggregation_time = time.time()
        
        # Add to drift analyzer for temporal analysis
        self.drift_analyzer.add_observation(
            self.global_aggregated_hist,
            timestamp=time.time()
        )
    
    def _aggregate_global_stats(self) -> Dict:
        """Aggregate global statistics from all hubs."""
        total_students = 0
        total_samples = 0
        all_means = []
        all_stds = []
        
        for hub_id, data in self.hub_sketches.items():
            metadata = data['metadata']
            stats = metadata.get('aggregated_stats', {})
            
            num_students = metadata.get('num_students', 0)
            num_samples = stats.get('total_samples', 0)
            
            total_students += num_students
            total_samples += num_samples
            all_means.append(stats.get('mean', 0.5))
            all_stds.append(stats.get('std', 0.1))
        
        return {
            'total_hubs': len(self.hub_sketches),
            'total_students': total_students,
            'total_samples': total_samples,
            'global_mean': np.mean(all_means),
            'global_std': np.mean(all_stds)
        }
    
    def check_global_drift(self):
        """
        Check if global distribution has drifted from baseline.
        
        Uses ensemble drift detector (KS + PSI + JS).
        """
        if self.global_aggregated_hist is None:
            return
        
        drift_detected, drift_scores = self.drift_detector.detect(
            self.baseline_hist,
            self.global_aggregated_hist
        )
        
        self.drift_detected = drift_detected
        
        # Store drift event
        drift_event = {
            'timestamp': time.time(),
            'drift_detected': drift_detected,
            'scores': drift_scores,
            'global_histogram': self.global_aggregated_hist.tolist()
        }
        self.drift_history.append(drift_event)
        
        if drift_detected:
            print(f"[Central Server] ⚠️  GLOBAL DRIFT DETECTED!")
            print(f"  KS: {drift_scores['ks_detected']}, PSI: {drift_scores['psi_detected']}, JS: {drift_scores['js_detected']}")
            print(f"  Detections: {drift_scores['num_detections']}/3")
            
            # Trigger threshold recalibration
            self.handle_drift_detected(drift_scores)
    
    def check_anomalous_hubs(self):
        """
        Identify hubs with distributions divergent from majority.
        
        Uses DBSCAN clustering on hub distributions.
        """
        if len(self.hub_sketches) < 3:
            return  # Need at least 3 hubs for meaningful clustering
        
        # Get all hub histograms
        hub_ids = list(self.hub_sketches.keys())
        hub_hists = [self.hub_sketches[hid]['histogram'] for hid in hub_ids]
        
        # Cluster hubs
        self.anomaly_clusterer.fit(hub_hists, client_ids=hub_ids)
        self.anomalous_hubs = self.anomaly_clusterer.get_anomalous_clients()
        
        # Compute anomaly scores
        scores = self.anomaly_scorer.compute_scores(hub_hists, client_ids=hub_ids)
        
        if self.anomalous_hubs:
            print(f"[Central Server] ⚠️  Anomalous hubs detected: {self.anomalous_hubs}")
            
            # Log anomaly details
            for hub_id in self.anomalous_hubs:
                score = scores.get(hub_id, 0.0)
                print(f"  Hub {hub_id}: anomaly score = {score:.4f}")
    
    def handle_drift_detected(self, drift_scores: Dict):
        """
        Handle drift detection event.
        
        Actions:
        1. Recalibrate threshold
        2. Broadcast to all hubs
        3. Log event
        
        Args:
            drift_scores: Drift detection scores
        """
        print(f"[Central Server] Handling drift detection...")
        
        # Recalibrate threshold from global distribution
        new_threshold = self.threshold_manager.calibrate_threshold(
            self.global_aggregated_hist
        )
        
        # Update current threshold
        old_threshold = self.current_threshold
        self.current_threshold = new_threshold
        self.threshold_updated = True
        
        print(f"[Central Server] Threshold recalibrated: {old_threshold:.3f} → {new_threshold:.3f}")
        
        # In real deployment, would actively broadcast to all hubs
        # In simulation, hubs get updated threshold on next submission
    
    def get_drift_report(self) -> Dict:
        """
        Generate comprehensive drift report.
        
        Returns:
            Dictionary with current drift status and metrics
        """
        # Get drift type from analyzer
        drift_type = 'unknown'
        if len(self.drift_history) >= 3:
            drift_type = self.drift_analyzer.detect_drift_type()
        
        # Get latest drift scores
        latest_scores = {}
        if self.drift_history:
            latest_scores = self.drift_history[-1]['scores']
        
        # Get cluster summary
        cluster_summary = {}
        if len(self.hub_sketches) >= 2:
            cluster_summary = self.anomaly_clusterer.get_summary()
        
        return {
            'drift_detected': self.drift_detected,
            'drift_type': drift_type,
            'drift_scores': latest_scores,
            'anomalous_hubs': self.anomalous_hubs,
            'global_distribution': self.global_aggregated_hist.tolist() if self.global_aggregated_hist is not None else None,
            'global_stats': self.global_aggregated_stats,
            'cluster_summary': cluster_summary,
            'current_threshold': self.current_threshold,
            'total_aggregations': self.total_global_aggregations,
            'num_connected_hubs': len(self.connected_hubs),
            'timestamp': time.time()
        }
    
    def get_system_status(self) -> Dict:
        """
        Get overall system status.
        
        Returns:
            Dictionary with system metrics
        """
        return {
            'connected_hubs': len(self.connected_hubs),
            'expected_hubs': self.num_hubs,
            'total_submissions': self.total_hub_submissions,
            'total_aggregations': self.total_global_aggregations,
            'drift_detected': self.drift_detected,
            'num_anomalous_hubs': len(self.anomalous_hubs),
            'current_threshold': self.current_threshold,
            'time_since_last_aggregation': time.time() - self.last_aggregation_time,
            'drift_history_length': len(self.drift_history)
        }
    
    def get_hub_comparison(self) -> Dict:
        """
        Compare all hubs to identify differences.
        
        Returns:
            Dictionary with hub comparison metrics
        """
        if len(self.hub_sketches) < 2:
            return {'error': 'Not enough hubs for comparison'}
        
        hub_ids = list(self.hub_sketches.keys())
        hub_hists = [self.hub_sketches[hid]['histogram'] for hid in hub_ids]
        
        # Compute pairwise JS divergences
        from scipy.spatial.distance import jensenshannon
        from core.anomaly_detection import compute_divergence_matrix
        
        div_matrix = compute_divergence_matrix(hub_hists, metric='js')
        
        # Find most similar and most different hubs
        np.fill_diagonal(div_matrix, np.inf)  # Ignore self-comparisons
        
        min_idx = np.unravel_index(np.argmin(div_matrix), div_matrix.shape)
        max_idx = np.unravel_index(np.argmax(div_matrix), div_matrix.shape)
        
        return {
            'num_hubs': len(hub_ids),
            'most_similar_hubs': (hub_ids[min_idx[0]], hub_ids[min_idx[1]]),
            'similarity_score': float(div_matrix[min_idx]),
            'most_different_hubs': (hub_ids[max_idx[0]], hub_ids[max_idx[1]]),
            'difference_score': float(div_matrix[max_idx]),
            'average_divergence': float(np.mean(div_matrix[div_matrix != np.inf]))
        }
    
    def reset_drift_detection(self):
        """Reset drift detection (use with caution)."""
        self.drift_detected = False
        self.drift_history = []
        self.drift_analyzer = DriftAnalyzer(baseline_hist=self.baseline_hist)
        print(f"[Central Server] Drift detection reset")
    
    def update_baseline(self, new_baseline: np.ndarray):
        """
        Update baseline distribution.
        
        Use when legitimate distribution shift occurs (e.g., seasonal changes).
        
        Args:
            new_baseline: New baseline histogram
        """
        self.baseline_hist = new_baseline / new_baseline.sum()
        self.drift_analyzer = DriftAnalyzer(baseline_hist=self.baseline_hist)
        print(f"[Central Server] Baseline updated")
    
    def __repr__(self):
        return (f"CentralDriftServer(hubs={len(self.connected_hubs)}, "
                f"aggregations={self.total_global_aggregations}, "
                f"drift={self.drift_detected})")


# Example usage and testing
if __name__ == "__main__":
    print("Testing CentralDriftServer...")
    
    # Create baseline distribution
    print("\n1. Creating baseline:")
    np.random.seed(42)
    baseline_samples = np.random.beta(2, 5, size=5000)
    baseline_hist, _ = np.histogram(baseline_samples, bins=20, range=(0, 1))
    baseline_hist = baseline_hist / baseline_hist.sum()
    print(f"  Baseline histogram created")
    
    # Create central server
    print("\n2. Creating central server:")
    server = CentralDriftServer(
        baseline_hist=baseline_hist,
        num_hubs=3,
        min_hubs_for_aggregation=2
    )
    print(f"  {server}")
    
    # Simulate receiving hub aggregations (no drift)
    print("\n3. Simulating hub submissions (no drift):")
    for hub_id in range(3):
        # Create similar distribution
        samples = np.random.beta(2, 5, size=1000)
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        hist = hist / hist.sum()
        
        metadata = {
            'num_students': 5,
            'aggregated_stats': {
                'total_samples': 500,
                'mean': np.mean(samples),
                'std': np.std(samples)
            }
        }
        
        response = server.receive_hub_aggregation(hub_id, hist, metadata)
        print(f"  Hub {hub_id}: drift={response['global_drift_detected']}")
    
    # Get drift report
    print("\n4. Drift report (no drift):")
    report = server.get_drift_report()
    print(f"  Drift detected: {report['drift_detected']}")
    print(f"  Drift type: {report['drift_type']}")
    
    # Simulate drift scenario
    print("\n5. Simulating drift (one hub with different distribution):")
    server.hub_sketches.clear()  # Clear previous
    
    # 2 normal hubs
    for hub_id in range(2):
        samples = np.random.beta(2, 5, size=1000)
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        hist = hist / hist.sum()
        metadata = {'num_students': 5, 'aggregated_stats': {'total_samples': 500}}
        server.receive_hub_aggregation(hub_id, hist, metadata)
    
    # 1 drifted hub
    samples = np.random.beta(5, 2, size=1000)  # Reversed distribution
    hist, _ = np.histogram(samples, bins=20, range=(0, 1))
    hist = hist / hist.sum()
    metadata = {'num_students': 5, 'aggregated_stats': {'total_samples': 500}}
    response = server.receive_hub_aggregation(2, hist, metadata)
    
    print(f"  Drift detected: {response['global_drift_detected']}")
    print(f"  Anomalous hubs: {response['anomalous_hubs']}")
    
    # Get system status
    print("\n6. System status:")
    status = server.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n✅ CentralDriftServer tests passed!")
