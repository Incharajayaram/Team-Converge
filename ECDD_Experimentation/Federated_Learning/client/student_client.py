"""
Student client for hierarchical federated learning.

Wraps the student model (Tiny LaDeDa) with federated monitoring.
Sends sketches to local hub (Raspberry Pi with teacher model).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import time
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.client_monitor import ClientMonitor


class StudentClient:
    """
    Client running student model (Arduino Nicla) with federated monitoring.
    
    Communicates with local hub (Raspberry Pi) rather than central server.
    """
    
    def __init__(self,
                 student_model: nn.Module,
                 hub_url: str,
                 client_id: int,
                 window_size: int = 500,
                 epsilon: float = 1.0,
                 initial_threshold: float = 0.5,
                 baseline_hist: Optional[np.ndarray] = None):
        """
        Initialize student client.
        
        Args:
            student_model: PyTorch student model (Tiny LaDeDa)
            hub_url: URL of local hub (e.g., 'http://192.168.1.10:6000')
            client_id: Unique client identifier
            window_size: Predictions before sending sketch
            epsilon: Privacy parameter
            initial_threshold: Initial classification threshold
            baseline_hist: Optional baseline for local drift detection
        """
        self.model = student_model
        self.model.eval()  # Inference mode
        
        self.hub_url = hub_url
        self.client_id = client_id
        self.threshold = initial_threshold
        
        # Initialize monitor
        self.monitor = ClientMonitor(
            window_size=window_size,
            num_bins=20,
            epsilon=epsilon,
            track_quantiles=True,
            local_drift_detection=baseline_hist is not None,
            baseline_hist=baseline_hist
        )
        
        # Performance tracking
        self.inference_times = []
        self.communication_times = []
        self.last_threshold_update = time.time()
        
        # Connection status
        self.hub_reachable = self._check_hub_connection()
    
    def _check_hub_connection(self) -> bool:
        """Check if hub is reachable."""
        try:
            response = requests.get(f"{self.hub_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Run inference and update monitoring.
        
        Args:
            image: Input image tensor (preprocessed)
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Inference
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Move to device (CPU for Arduino, GPU for simulation)
        device = next(self.model.parameters()).device
        image = image.to(device)
        
        # Forward pass
        output = self.model(image)
        
        # Get score
        if isinstance(output, tuple):
            logits = output[0]  # Handle multi-output models
        else:
            logits = output
        
        score = torch.sigmoid(logits).item()
        
        # Compute confidence (softmax entropy or margin)
        confidence = self._compute_confidence(logits)
        
        # Detect OOD (simplified - use actual OOD detector if available)
        is_ood = confidence < 0.6
        
        # Abstain if very uncertain
        abstained = confidence < 0.5
        
        # Make prediction
        prediction = score > self.threshold and not abstained
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Update monitor
        self.monitor.update(
            score=score,
            confidence=confidence,
            is_ood=is_ood,
            abstained=abstained
        )
        
        # Check if should send sketch to hub
        if self.monitor.is_buffer_full():
            self.send_sketch_to_hub()
        
        return {
            'prediction': prediction,
            'score': score,
            'confidence': confidence,
            'is_ood': is_ood,
            'abstained': abstained,
            'threshold': self.threshold,
            'inference_time': inference_time
        }
    
    def _compute_confidence(self, logits: torch.Tensor) -> float:
        """
        Compute prediction confidence.
        
        Uses distance from decision boundary as proxy for confidence.
        
        Args:
            logits: Model logits
            
        Returns:
            Confidence score (0-1)
        """
        # Simple approach: distance from 0 (decision boundary)
        # More sophisticated: use temperature-scaled softmax or MC dropout
        score = torch.sigmoid(logits).item()
        
        # Distance from 0.5 (decision boundary), normalized
        confidence = 2 * abs(score - 0.5)  # 0 at boundary, 1 at extremes
        
        return min(confidence, 1.0)
    
    def send_sketch_to_hub(self):
        """
        Send privacy-preserving sketch to local hub.
        
        This is called automatically when buffer is full.
        """
        start_time = time.time()
        
        try:
            # Generate sketch
            sketch = self.monitor.get_sketch(apply_dp=True, clear_after=True)
            
            # Add client metadata
            payload = {
                'client_id': self.client_id,
                'sketch': sketch,
                'client_type': 'student',
                'timestamp': time.time()
            }
            
            # Send to hub
            if self.hub_reachable:
                response = requests.post(
                    f"{self.hub_url}/submit_sketch",
                    json=payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Check for threshold update
                    if 'threshold_update' in response_data:
                        self._update_threshold(response_data['threshold_update'])
                    
                    # Track communication time
                    comm_time = time.time() - start_time
                    self.communication_times.append(comm_time)
                    
                    return True
            else:
                print(f"[Client {self.client_id}] Hub unreachable, sketch not sent")
                return False
                
        except Exception as e:
            print(f"[Client {self.client_id}] Error sending sketch: {e}")
            return False
    
    def _update_threshold(self, new_threshold: float):
        """
        Update classification threshold from hub.
        
        Args:
            new_threshold: New threshold value
        """
        old_threshold = self.threshold
        self.threshold = new_threshold
        self.last_threshold_update = time.time()
        
        print(f"[Client {self.client_id}] Threshold updated: {old_threshold:.3f} → {new_threshold:.3f}")
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.inference_times) == 0:
            return {'no_data': True}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'total_predictions': len(self.inference_times),
            'avg_communication_time': np.mean(self.communication_times) if self.communication_times else 0.0,
            'total_sketch_sends': len(self.communication_times),
            'current_threshold': self.threshold,
            'time_since_last_threshold_update': time.time() - self.last_threshold_update,
            'monitor_stats': self.monitor.get_current_stats()
        }
    
    def __repr__(self):
        return f"StudentClient(id={self.client_id}, hub={self.hub_url}, predictions={len(self.inference_times)})"


class MockStudentModel(nn.Module):
    """
    Mock student model for testing (replace with actual Tiny LaDeDa).
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Example usage and testing
if __name__ == "__main__":
    print("Testing StudentClient...")
    
    # Create mock student model
    print("\n1. Creating mock student model:")
    model = MockStudentModel()
    model.eval()
    print(f"  Model: {model.__class__.__name__}")
    
    # Create student client (without actual hub connection)
    print("\n2. Creating student client:")
    client = StudentClient(
        student_model=model,
        hub_url="http://localhost:6000",  # Mock hub
        client_id=1,
        window_size=10,  # Small for testing
        epsilon=1.0
    )
    print(f"  {client}")
    print(f"  Hub reachable: {client.hub_reachable}")
    
    # Simulate predictions
    print("\n3. Simulating predictions:")
    np.random.seed(42)
    
    for i in range(15):
        # Create random image
        image = torch.randn(1, 3, 64, 64)
        
        # Predict
        result = client.predict(image)
        
        if i % 5 == 0:
            print(f"  Prediction {i}: score={result['score']:.3f}, conf={result['confidence']:.3f}, pred={result['prediction']}")
    
    # Get performance stats
    print("\n4. Performance statistics:")
    stats = client.get_performance_stats()
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Avg inference time: {stats['avg_inference_time']*1000:.2f} ms")
    print(f"  Current threshold: {stats['current_threshold']:.3f}")
    print(f"  Monitor buffer: {stats['monitor_stats']['buffer_size']}/{stats['monitor_stats']['buffer_full']}")
    
    # Test manual sketch generation
    print("\n5. Manual sketch generation:")
    if client.monitor.is_buffer_full():
        sketch = client.monitor.get_sketch(apply_dp=True, clear_after=False)
        print(f"  Sketch metadata: {sketch['metadata']}")
        print(f"  Privacy: ε={sketch['privacy']['epsilon']}, remaining={sketch['privacy']['remaining_budget']:.2f}")
    else:
        print(f"  Buffer not full yet ({client.monitor.get_current_stats()['buffer_size']} samples)")
    
    print("\n✅ StudentClient tests passed!")
    print("\nNote: Replace MockStudentModel with your actual Tiny LaDeDa model")
    print("      from Team-Converge/deepfake-patch-audit/models/student/tiny_ladeda.py")
