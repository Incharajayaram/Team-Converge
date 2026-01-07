"""
REST API server for federated drift detection.

Provides endpoints for:
- Hub aggregation submission
- Status queries
- Threshold queries
- Drift reports
"""

from flask import Flask, request, jsonify
import numpy as np
from typing import Dict, Optional
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.drift_server import CentralDriftServer
from server.teacher_aggregator import TeacherHubAggregator

# Initialize Flask app
app = Flask(__name__)

# Global server instances (will be initialized in main)
central_server: Optional[CentralDriftServer] = None
hub_servers: Dict[int, TeacherHubAggregator] = {}


# ==================== Central Server Endpoints ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'federated_drift_server'}), 200


@app.route('/submit_hub_aggregation', methods=['POST'])
def submit_hub_aggregation():
    """
    Receive aggregated sketch from a hub.
    
    Expected payload:
    {
        'hub_id': int,
        'aggregated_histogram': list[float],
        'aggregated_stats': dict,
        'num_students': int,
        'local_drift_detected': bool,
        'timestamp': float
    }
    """
    if central_server is None:
        return jsonify({'error': 'Central server not initialized'}), 500
    
    try:
        data = request.json
        
        # Extract data
        hub_id = data['hub_id']
        aggregated_hist = np.array(data['aggregated_histogram'])
        metadata = {
            'num_students': data.get('num_students', 0),
            'aggregated_stats': data.get('aggregated_stats', {}),
            'local_drift_detected': data.get('local_drift_detected', False)
        }
        
        # Process with central server
        response = central_server.receive_hub_aggregation(hub_id, aggregated_hist, metadata)
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/get_status', methods=['GET'])
def get_status():
    """Get current system status."""
    if central_server is None:
        return jsonify({'error': 'Central server not initialized'}), 500
    
    try:
        status = central_server.get_system_status()
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_threshold', methods=['GET'])
def get_threshold():
    """Get current classification threshold."""
    if central_server is None:
        return jsonify({'error': 'Central server not initialized'}), 500
    
    return jsonify({'threshold': central_server.current_threshold}), 200


@app.route('/get_drift_report', methods=['GET'])
def get_drift_report():
    """Get comprehensive drift report."""
    if central_server is None:
        return jsonify({'error': 'Central server not initialized'}), 500
    
    try:
        report = central_server.get_drift_report()
        return jsonify(report), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_hub_comparison', methods=['GET'])
def get_hub_comparison():
    """Get comparison between hubs."""
    if central_server is None:
        return jsonify({'error': 'Central server not initialized'}), 500
    
    try:
        comparison = central_server.get_hub_comparison()
        return jsonify(comparison), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Hub Server Endpoints ====================

@app.route('/hub/<int:hub_id>/submit_sketch', methods=['POST'])
def submit_student_sketch(hub_id: int):
    """
    Receive sketch from a student client.
    
    Expected payload:
    {
        'client_id': int,
        'sketch': dict,
        'client_type': str,
        'timestamp': float
    }
    """
    if hub_id not in hub_servers:
        return jsonify({'error': f'Hub {hub_id} not found'}), 404
    
    try:
        data = request.json
        
        student_id = data['client_id']
        sketch = data['sketch']
        
        # Process with hub aggregator
        hub = hub_servers[hub_id]
        response = hub.receive_student_sketch(student_id, sketch)
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/hub/<int:hub_id>/get_stats', methods=['GET'])
def get_hub_stats(hub_id: int):
    """Get statistics for a specific hub."""
    if hub_id not in hub_servers:
        return jsonify({'error': f'Hub {hub_id} not found'}), 404
    
    try:
        hub = hub_servers[hub_id]
        stats = hub.get_hub_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/hub/<int:hub_id>/health', methods=['GET'])
def hub_health_check(hub_id: int):
    """Health check for specific hub."""
    if hub_id not in hub_servers:
        return jsonify({'error': f'Hub {hub_id} not found'}), 404
    
    return jsonify({'status': 'healthy', 'hub_id': hub_id}), 200


# ==================== Admin/Debug Endpoints ====================

@app.route('/admin/reset_drift', methods=['POST'])
def reset_drift():
    """Reset drift detection (admin only)."""
    if central_server is None:
        return jsonify({'error': 'Central server not initialized'}), 500
    
    try:
        central_server.reset_drift_detection()
        return jsonify({'status': 'drift_detection_reset'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/update_baseline', methods=['POST'])
def update_baseline():
    """Update baseline distribution (admin only)."""
    if central_server is None:
        return jsonify({'error': 'Central server not initialized'}), 500
    
    try:
        data = request.json
        new_baseline = np.array(data['baseline_histogram'])
        central_server.update_baseline(new_baseline)
        return jsonify({'status': 'baseline_updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/admin/list_hubs', methods=['GET'])
def list_hubs():
    """List all registered hubs."""
    return jsonify({
        'hubs': list(hub_servers.keys()),
        'num_hubs': len(hub_servers)
    }), 200


# ==================== Initialization Functions ====================

def initialize_central_server(baseline_hist: np.ndarray,
                              num_hubs: int,
                              min_hubs: int = 2,
                              target_fpr: float = 0.01):
    """
    Initialize central server.
    
    Args:
        baseline_hist: Baseline distribution
        num_hubs: Expected number of hubs
        min_hubs: Minimum hubs for aggregation
        target_fpr: Target false positive rate
    """
    global central_server
    
    central_server = CentralDriftServer(
        baseline_hist=baseline_hist,
        num_hubs=num_hubs,
        min_hubs_for_aggregation=min_hubs,
        target_fpr=target_fpr
    )
    
    print(f"✓ Central server initialized for {num_hubs} hubs")


def register_hub_server(hub_id: int, hub: TeacherHubAggregator):
    """
    Register a hub server.
    
    Args:
        hub_id: Hub identifier
        hub: TeacherHubAggregator instance
    """
    global hub_servers
    
    hub_servers[hub_id] = hub
    print(f"✓ Hub {hub_id} registered")


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """
    Run Flask server.
    
    Args:
        host: Host address
        port: Port number
        debug: Debug mode
    """
    print(f"\n{'='*60}")
    print(f"Starting Federated Drift Detection Server")
    print(f"{'='*60}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Central Server: {'Initialized' if central_server else 'Not initialized'}")
    print(f"Registered Hubs: {len(hub_servers)}")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug)


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("Federated Drift Detection Server - Test Mode\n")
    
    # Create baseline distribution
    print("1. Creating baseline distribution...")
    np.random.seed(42)
    baseline_samples = np.random.beta(2, 5, size=5000)
    baseline_hist, _ = np.histogram(baseline_samples, bins=20, range=(0, 1))
    baseline_hist = baseline_hist / baseline_hist.sum()
    print("   ✓ Baseline created")
    
    # Initialize central server
    print("\n2. Initializing central server...")
    initialize_central_server(
        baseline_hist=baseline_hist,
        num_hubs=3,
        min_hubs=2,
        target_fpr=0.01
    )
    
    # Note: In real deployment, hub servers would be on separate Raspberry Pi devices
    # For testing, we can create mock hubs
    print("\n3. Note: Hub servers would run on separate devices (Raspberry Pi)")
    print("   In simulation, hubs communicate via REST API\n")
    
    # Run server
    print("4. Starting Flask server...")
    print("   Available endpoints:")
    print("   - POST /submit_hub_aggregation")
    print("   - GET  /get_status")
    print("   - GET  /get_threshold")
    print("   - GET  /get_drift_report")
    print("   - GET  /health")
    print("\n   Press Ctrl+C to stop\n")
    
    try:
        run_server(host='127.0.0.1', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
