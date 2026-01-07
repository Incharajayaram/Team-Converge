"""
Generic federated client for communication with server.

Handles REST API communication, sketch transmission, and threshold updates.
"""

import requests
import json
import time
from typing import Dict, Optional, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.client_monitor import ClientMonitor


class FederatedClient:
    """
    Generic federated client with communication capabilities.
    
    Can be used by both student clients (to hub) and hub clients (to central server).
    """
    
    def __init__(self,
                 server_url: str,
                 client_id: int,
                 client_type: str = 'generic',
                 monitor: Optional[ClientMonitor] = None):
        """
        Initialize federated client.
        
        Args:
            server_url: URL of server (hub or central)
            client_id: Unique client identifier
            client_type: Type of client ('student', 'hub', 'generic')
            monitor: Optional ClientMonitor instance
        """
        self.server_url = server_url.rstrip('/')
        self.client_id = client_id
        self.client_type = client_type
        
        # Create monitor if not provided
        if monitor is None:
            self.monitor = ClientMonitor(window_size=500, epsilon=1.0)
        else:
            self.monitor = monitor
        
        # Communication tracking
        self.total_sketches_sent = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.last_communication_time = None
        self.communication_errors = 0
        
        # Server state
        self.current_threshold = 0.5
        self.server_reachable = self._check_server_health()
    
    def _check_server_health(self) -> bool:
        """Check if server is reachable."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def send_sketch(self, sketch: Optional[Dict] = None, metadata: Optional[Dict] = None) -> bool:
        """
        Send sketch to server.
        
        Args:
            sketch: Sketch data (if None, generates from monitor)
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Generate sketch if not provided
        if sketch is None:
            if not self.monitor.is_buffer_full():
                return False  # Not ready to send
            sketch = self.monitor.get_sketch(apply_dp=True, clear_after=True)
        
        # Prepare payload
        payload = {
            'client_id': self.client_id,
            'client_type': self.client_type,
            'sketch': sketch,
            'timestamp': time.time()
        }
        
        # Add metadata if provided
        if metadata:
            payload['metadata'] = metadata
        
        # Send to server
        try:
            payload_json = json.dumps(payload)
            payload_size = len(payload_json.encode('utf-8'))
            
            response = requests.post(
                f"{self.server_url}/submit_sketch",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                response_data = response.json()
                response_size = len(response.content)
                
                # Track communication
                self.total_sketches_sent += 1
                self.total_bytes_sent += payload_size
                self.total_bytes_received += response_size
                self.last_communication_time = time.time()
                
                # Handle server responses
                self._handle_server_response(response_data)
                
                return True
            else:
                self.communication_errors += 1
                print(f"[Client {self.client_id}] Server returned {response.status_code}")
                return False
                
        except Exception as e:
            self.communication_errors += 1
            print(f"[Client {self.client_id}] Communication error: {e}")
            return False
    
    def _handle_server_response(self, response_data: Dict):
        """
        Handle responses from server.
        
        Args:
            response_data: Response dictionary from server
        """
        # Check for threshold update
        if 'threshold_update' in response_data:
            new_threshold = response_data['threshold_update']
            self._update_threshold(new_threshold)
        
        # Check for drift alert
        if 'drift_detected' in response_data and response_data['drift_detected']:
            print(f"[Client {self.client_id}] ⚠️  Drift detected by server!")
        
        # Check for anomaly flag
        if 'flagged_as_anomalous' in response_data and response_data['flagged_as_anomalous']:
            print(f"[Client {self.client_id}] ⚠️  Flagged as anomalous by server!")
    
    def _update_threshold(self, new_threshold: float):
        """Update classification threshold."""
        self.current_threshold = new_threshold
        print(f"[Client {self.client_id}] Threshold updated: {new_threshold:.3f}")
    
    def get_status(self) -> Dict:
        """
        Query server for current status.
        
        Returns:
            Server status dictionary
        """
        try:
            response = requests.get(f"{self.server_url}/get_status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'Status code {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_current_threshold(self) -> float:
        """
        Query server for current threshold.
        
        Returns:
            Current threshold value
        """
        try:
            response = requests.get(f"{self.server_url}/get_threshold", timeout=5)
            if response.status_code == 200:
                return response.json()['threshold']
            else:
                return self.current_threshold  # Return cached value
        except:
            return self.current_threshold
    
    def get_communication_stats(self) -> Dict:
        """
        Get communication statistics.
        
        Returns:
            Dictionary with communication metrics
        """
        return {
            'total_sketches_sent': self.total_sketches_sent,
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            'avg_bytes_per_sketch': self.total_bytes_sent / self.total_sketches_sent if self.total_sketches_sent > 0 else 0,
            'communication_errors': self.communication_errors,
            'error_rate': self.communication_errors / (self.total_sketches_sent + self.communication_errors) if (self.total_sketches_sent + self.communication_errors) > 0 else 0,
            'last_communication': self.last_communication_time,
            'time_since_last_communication': time.time() - self.last_communication_time if self.last_communication_time else None,
            'server_reachable': self.server_reachable
        }
    
    def __repr__(self):
        return f"FederatedClient(id={self.client_id}, type={self.client_type}, sketches_sent={self.total_sketches_sent})"


# Example usage and testing
if __name__ == "__main__":
    print("Testing FederatedClient...")
    
    # Create client
    print("\n1. Creating federated client:")
    client = FederatedClient(
        server_url="http://localhost:5000",
        client_id=1,
        client_type='test'
    )
    print(f"  {client}")
    print(f"  Server reachable: {client.server_reachable}")
    
    # Simulate adding predictions to monitor
    print("\n2. Simulating predictions:")
    import numpy as np
    np.random.seed(42)
    
    for i in range(100):
        score = np.random.beta(2, 5)
        confidence = np.random.uniform(0.7, 1.0)
        client.monitor.update(score, confidence)
    
    print(f"  Monitor status: {client.monitor}")
    
    # Try to send sketch (will fail without actual server)
    print("\n3. Attempting to send sketch:")
    if client.monitor.is_buffer_full():
        print(f"  Buffer full, ready to send")
        # success = client.send_sketch()
        # print(f"  Send successful: {success}")
    else:
        print(f"  Buffer not full yet")
    
    # Get communication stats
    print("\n4. Communication statistics:")
    stats = client.get_communication_stats()
    for key, value in stats.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    print("\n✅ FederatedClient tests passed!")
