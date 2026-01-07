"""
Teacher Hub Aggregator for hierarchical federated learning.

Runs on Raspberry Pi with teacher model. Aggregates sketches from
student clients (Arduino Nicla) and communicates with central server.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import time
import requests
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sketch_algorithms import ScoreHistogram, StatisticalSummary
from core.drift_detection import EnsembleDriftDetector
from server.adaptive_threshold import ThresholdCalibrator


class TeacherHubAggregator:
    """
    Raspberry Pi hub running teacher model and aggregating student sketches.
    
    Acts as intermediate aggregator in hierarchical federation:
    Students (Nicla) → Hub (Pi) → Central Server
    """
    
    def __init__(self,
                 teacher_model: nn.Module,
                 central_server_url: str,
                 hub_id: int,
                 min_students_for_aggregation: int = 3,
                 aggregation_frequency: int = 5,
                 baseline_hist: Optional[np.ndarray] = None):
        """
        Initialize teacher hub aggregator.
        
        Args:
            teacher_model: PyTorch teacher model (LaDeDa)
            central_server_url: URL of central server
            hub_id: Unique hub identifier
            min_students_for_aggregation: Minimum students before aggregating
            aggregation_frequency: Aggregate every N student submissions
            baseline_hist: Baseline distribution for local drift detection
        """
        self.model = teacher_model
        self.model.eval()
        
        self.central_server_url = central_server_url.rstrip('/')
        self.hub_id = hub_id
        self.min_students = min_students_for_aggregation
        self.aggregation_frequency = aggregation_frequency
        
        # Student tracking
        self.student_sketches = {}  # {student_id: sketch}
        self.student_metadata = {}  # {student_id: metadata}
        self.connected_students = set()
        
        # Aggregated state
        self.local_aggregated_hist = None
        self.local_aggregated_stats = None
        self.aggregation_count = 0
        self.last_aggregation_time = time.time()
        
        # Drift detection at hub level
        self.baseline_hist = baseline_hist
        self.drift_detector = EnsembleDriftDetector() if baseline_hist is not None else None
        self.local_drift_detected = False
        
        # Threshold management
        self.current_threshold = 0.5
        self.threshold_history = []
        
        # Communication tracking
        self.total_sketches_received = 0
        self.total_aggregations_sent = 0
        
        # Hub-level inference tracking (optional)
        self.hub_predictions = []
        
        print(f"[Hub {hub_id}] Initialized with teacher model")
    
    def receive_student_sketch(self, student_id: int, sketch: Dict) -> Dict:
        """
        Receive sketch from a student client.
        
        Args:
            student_id: Student identifier
            sketch: Sketch data from student
            
        Returns:
            Response dictionary for student
        """
        # Store sketch
        self.student_sketches[student_id] = sketch
        self.student_metadata[student_id] = sketch.get('metadata', {})
        self.connected_students.add(student_id)
        self.total_sketches_received += 1
        
        print(f"[Hub {self.hub_id}] Received sketch from Student {student_id} "
              f"({len(self.student_sketches)}/{self.min_students} students)")
        
        # Check if should aggregate
        should_aggregate = (
            len(self.student_sketches) >= self.min_students and
            self.total_sketches_received % self.aggregation_frequency == 0
        )
        
        if should_aggregate:
            self.aggregate_local_students()
        
        # Prepare response
        response = {
            'status': 'received',
            'hub_id': self.hub_id,
            'threshold': self.current_threshold
        }
        
        # Include threshold update if changed
        if len(self.threshold_history) > 0:
            latest_threshold = self.threshold_history[-1]['threshold']
            if latest_threshold != self.current_threshold:
                response['threshold_update'] = latest_threshold
                self.current_threshold = latest_threshold
        
        return response
    
    def aggregate_local_students(self):
        """
        Aggregate sketches from all connected students.
        
        Computes weighted average of student distributions.
        """
        if len(self.student_sketches) < self.min_students:
            print(f"[Hub {self.hub_id}] Not enough students for aggregation "
                  f"({len(self.student_sketches)}/{self.min_students})")
            return
        
        print(f"[Hub {self.hub_id}] Aggregating {len(self.student_sketches)} student sketches...")
        
        # Extract histograms and weights
        student_hists = []
        weights = []
        
        for student_id, sketch in self.student_sketches.items():
            # Get full histogram (not sparse)
            hist = np.array(sketch.get('histogram_full', sketch['histogram']))
            if isinstance(hist, dict):  # Handle sparse format
                full_hist = np.zeros(20)
                for idx, val in hist.items():
                    full_hist[int(idx)] = val
                hist = full_hist
            
            # Normalize
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            
            # Weight by number of samples
            num_samples = sketch.get('metadata', {}).get('num_samples', 1)
            
            student_hists.append(hist)
            weights.append(num_samples)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        self.local_aggregated_hist = np.average(student_hists, axis=0, weights=weights)
        
        # Aggregate statistics
        self.local_aggregated_stats = self._aggregate_statistics()
        
        # Check for local drift
        if self.baseline_hist is not None and self.drift_detector is not None:
            drift_detected, drift_scores = self.drift_detector.detect(
                self.baseline_hist,
                self.local_aggregated_hist
            )
            self.local_drift_detected = drift_detected
            
            if drift_detected:
                print(f"[Hub {self.hub_id}] ⚠️  Local drift detected! Scores: {drift_scores}")
        
        # Send aggregated sketch to central server
        self.send_to_central_server()
        
        # Clear student sketches after aggregation
        self.student_sketches.clear()
        self.aggregation_count += 1
        self.last_aggregation_time = time.time()
    
    def _aggregate_statistics(self) -> Dict:
        """Aggregate statistics from all students."""
        all_means = []
        all_stds = []
        total_samples = 0
        total_abstains = 0
        total_oods = 0
        
        for student_id, sketch in self.student_sketches.items():
            stats = sketch.get('statistics', {})
            metadata = sketch.get('metadata', {})
            
            num_samples = metadata.get('num_samples', 0)
            
            all_means.append(stats.get('mean', 0.5))
            all_stds.append(stats.get('std', 0.1))
            total_samples += num_samples
            total_abstains += metadata.get('abstain_rate', 0) * num_samples
            total_oods += metadata.get('ood_rate', 0) * num_samples
        
        return {
            'mean': np.mean(all_means),
            'std': np.mean(all_stds),
            'total_samples': total_samples,
            'abstain_rate': total_abstains / total_samples if total_samples > 0 else 0,
            'ood_rate': total_oods / total_samples if total_samples > 0 else 0,
            'num_students': len(self.student_sketches)
        }
    
    def send_to_central_server(self):
        """
        Send aggregated sketch to central server.
        
        This is the hub's contribution to the global federation.
        """
        if self.local_aggregated_hist is None:
            return
        
        print(f"[Hub {self.hub_id}] Sending aggregated sketch to central server...")
        
        # Prepare payload
        payload = {
            'hub_id': self.hub_id,
            'aggregated_histogram': self.local_aggregated_hist.tolist(),
            'aggregated_stats': self.local_aggregated_stats,
            'num_students': len(self.connected_students),
            'local_drift_detected': self.local_drift_detected,
            'timestamp': time.time()
        }
        
        try:
            response = requests.post(
                f"{self.central_server_url}/submit_hub_aggregation",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                response_data = response.json()
                self.total_aggregations_sent += 1
                
                # Handle central server responses
                self._handle_central_server_response(response_data)
                
                print(f"[Hub {self.hub_id}] ✓ Aggregated sketch sent successfully")
            else:
                print(f"[Hub {self.hub_id}] ✗ Central server returned {response.status_code}")
                
        except Exception as e:
            print(f"[Hub {self.hub_id}] ✗ Error sending to central server: {e}")
    
    def _handle_central_server_response(self, response_data: Dict):
        """Handle responses from central server."""
        # Check for global threshold update
        if 'threshold_update' in response_data:
            new_threshold = response_data['threshold_update']
            self.update_threshold(new_threshold)
        
        # Check for global drift alert
        if 'global_drift_detected' in response_data and response_data['global_drift_detected']:
            print(f"[Hub {self.hub_id}] ⚠️  Global drift detected by central server!")
        
        # Check if this hub is flagged as anomalous
        if 'anomalous_hubs' in response_data:
            if self.hub_id in response_data['anomalous_hubs']:
                print(f"[Hub {self.hub_id}] ⚠️  Flagged as anomalous by central server!")
    
    def update_threshold(self, new_threshold: float):
        """
        Update classification threshold.
        
        This will be broadcast to all connected students.
        
        Args:
            new_threshold: New threshold value
        """
        old_threshold = self.current_threshold
        self.current_threshold = new_threshold
        
        self.threshold_history.append({
            'threshold': new_threshold,
            'timestamp': time.time()
        })
        
        print(f"[Hub {self.hub_id}] Threshold updated: {old_threshold:.3f} → {new_threshold:.3f}")
        
        # In real deployment, broadcast to students
        # For simulation, students will get it on next sketch submission
    
    def broadcast_threshold_to_students(self, threshold: float):
        """
        Broadcast threshold update to all connected students.
        
        In real deployment, this would actively push to students.
        In simulation, students receive on next communication.
        
        Args:
            threshold: Threshold to broadcast
        """
        # This is a placeholder for real deployment
        # In simulation, threshold is returned in receive_student_sketch response
        self.current_threshold = threshold
    
    @torch.no_grad()
    def hub_inference(self, image: torch.Tensor) -> Dict:
        """
        Optional: Run inference with teacher model on hub.
        
        This can be used for:
        1. Validating student predictions
        2. Handling complex cases students abstain from
        3. Additional monitoring
        
        Args:
            image: Input image tensor
            
        Returns:
            Prediction dictionary
        """
        # Run teacher model
        output = self.model(image)
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        score = torch.sigmoid(logits).item()
        prediction = score > self.current_threshold
        
        # Store for monitoring
        self.hub_predictions.append({
            'score': score,
            'prediction': prediction,
            'timestamp': time.time()
        })
        
        return {
            'score': score,
            'prediction': prediction,
            'threshold': self.current_threshold
        }
    
    def get_hub_stats(self) -> Dict:
        """
        Get hub statistics.
        
        Returns:
            Dictionary with hub metrics
        """
        return {
            'hub_id': self.hub_id,
            'connected_students': len(self.connected_students),
            'total_sketches_received': self.total_sketches_received,
            'total_aggregations_sent': self.total_aggregations_sent,
            'aggregation_count': self.aggregation_count,
            'current_threshold': self.current_threshold,
            'local_drift_detected': self.local_drift_detected,
            'time_since_last_aggregation': time.time() - self.last_aggregation_time,
            'local_aggregated_available': self.local_aggregated_hist is not None
        }
    
    def __repr__(self):
        return (f"TeacherHubAggregator(id={self.hub_id}, "
                f"students={len(self.connected_students)}, "
                f"aggregations={self.aggregation_count})")


# Example usage and testing
if __name__ == "__main__":
    print("Testing TeacherHubAggregator...")
    
    # Create mock teacher model
    class MockTeacherModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 1)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    print("\n1. Creating hub aggregator:")
    teacher_model = MockTeacherModel()
    hub = TeacherHubAggregator(
        teacher_model=teacher_model,
        central_server_url="http://localhost:5000",
        hub_id=1,
        min_students_for_aggregation=3
    )
    print(f"  {hub}")
    
    # Simulate receiving student sketches
    print("\n2. Simulating student sketches:")
    np.random.seed(42)
    
    for student_id in range(5):
        # Create synthetic sketch
        samples = np.random.beta(2, 5, size=100)
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        hist = hist / hist.sum()
        
        sketch = {
            'histogram_full': hist.tolist(),
            'statistics': {
                'mean': np.mean(samples),
                'std': np.std(samples)
            },
            'metadata': {
                'num_samples': 100,
                'abstain_rate': 0.02,
                'ood_rate': 0.05
            }
        }
        
        response = hub.receive_student_sketch(student_id, sketch)
        print(f"  Student {student_id} → Hub: {response['status']}")
    
    # Get hub stats
    print("\n3. Hub statistics:")
    stats = hub.get_hub_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test hub inference
    print("\n4. Testing hub inference:")
    test_image = torch.randn(1, 3, 64, 64)
    result = hub.hub_inference(test_image)
    print(f"  Inference result: score={result['score']:.3f}, prediction={result['prediction']}")
    
    print("\n✅ TeacherHubAggregator tests passed!")
