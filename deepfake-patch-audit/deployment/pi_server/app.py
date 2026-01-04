#!/usr/bin/env python3
"""
Raspberry Pi Flask Server for Deepfake Detection
- Receives 96x96 JPEG images from Nicla Vision devices
- Runs TFLite/ONNX inference on Raspberry Pi
- Returns predictions with confidence scores
- Stores suspicious images and logs alerts
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from PIL import Image
import io
import logging
from threading import Thread, Lock
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import onnxruntime as rt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re
import time
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepfake_detections.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Flask-Limiter for rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],  # Default: 100 requests/hour per IP
    storage_uri="memory://"  # Use in-memory storage
)

# Enable CORS for all routes - allow cross-origin requests from any source
CORS(app,
     origins="*",
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False)

# Configuration
CONFIG = {
    'MODEL_PATH': 'models/deepfake_detector.onnx',
    'STORAGE_DIR': 'storage/suspicious_images',
    'STATS_FILE': 'storage/stats.json',
    'THRESHOLD': 0.84,  # Calibrated threshold
    'EMAIL_CONFIG': {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': '',
        'sender_password': '',
        'recipient_email': ''
    },
    'WEBHOOK_URL': None,  # Set if using webhook alerts
}

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Global variables
onnx_session = None
device_stats = {}
captured_images = {}  # Store captured image metadata from devices
stats_lock = Lock()  # ✅ PHASE 2.4: Thread-safe access to device_stats
cleanup_counter = 0  # ✅ PHASE 2.5: Trigger cleanup every 100 predictions


def initialize_model():
    """Load ONNX model using ONNX Runtime."""
    global onnx_session
    try:
        model_path = CONFIG['MODEL_PATH']
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return False

        # Use CPU provider on Raspberry Pi (GPU not available)
        onnx_session = rt.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        logger.info(f"✓ ONNX model loaded: {model_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load ONNX model: {e}")
        return False


def preprocess_image(image_data):
    """
    Preprocess 128x128 image from Nicla to 256x256 for inference.

    Pipeline:
    1. Load 128x128 JPEG from Nicla (~10-15 KB)
    2. Decompress to RGB bitmap
    3. Resize to 256x256 (2x upsampling with BICUBIC)
    4. Normalize with ImageNet mean/std
    5. Convert to (B, C, H, W) tensor format

    Args:
        image_data: Binary JPEG image data (128x128)

    Returns:
        (1, 3, 256, 256) numpy array in float32 format ready for ONNX inference
    """
    try:
        # Load JPEG image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Resize to 256x256
        image = image.resize((256, 256), Image.BICUBIC)

        # Convert to numpy array and normalize to [0, 1]
        image_np = np.array(image, dtype=np.float32) / np.float32(255.0)

        # Apply ImageNet normalization (ensure float32 throughout)
        mean = IMAGENET_MEAN.astype(np.float32)
        std = IMAGENET_STD.astype(np.float32)
        image_np = (image_np - mean) / std

        # Ensure float32 type
        image_np = image_np.astype(np.float32)

        # Convert to (C, H, W) then add batch dimension
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)

        # Final type check
        logger.info(f"Preprocessed image shape: {image_np.shape}, dtype: {image_np.dtype}")
        assert image_np.dtype == np.float32, f"Image dtype is {image_np.dtype}, expected float32"

        return image_np
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_inference(image_np):
    """
    Run inference using ONNX model.

    Args:
        image_np: (1, 3, 256, 256) numpy array in float32 format

    Returns:
        (patch_logits, is_fake, fake_prob, confidence)
    """
    try:
        logger.info(f"Running inference with input shape: {image_np.shape}, dtype: {image_np.dtype}")

        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name

        logger.info(f"Input name: {input_name}, Output name: {output_name}")

        # Run inference
        patch_logits = onnx_session.run(
            [output_name],
            {input_name: image_np}
        )[0]  # Shape: (1, 1, 126, 126)

        logger.info(f"Inference successful! Output shape: {patch_logits.shape}")

        # TopK Pooling: Select top 10% of patches
        batch_size, channels, h, w = patch_logits.shape
        num_patches = h * w
        k = max(5, int(np.ceil(0.1 * num_patches)))

        # Flatten patches
        patch_flat = patch_logits.reshape(batch_size, -1)  # (1, 15876)

        # Select top-k
        top_indices = np.argsort(patch_flat[0])[-k:]
        top_logits = patch_flat[0, top_indices]

        # Mean aggregation
        image_logit = np.mean(top_logits)

        # Sigmoid to probability
        fake_prob = 1.0 / (1.0 + np.exp(-image_logit))
        real_prob = 1.0 - fake_prob

        # Decision threshold
        is_fake = fake_prob > CONFIG['THRESHOLD']
        confidence = max(fake_prob, real_prob)

        return patch_logits, is_fake, float(fake_prob), float(confidence)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return None, False, 0.0, 0.0


def cleanup_old_images(max_storage_mb=1000, max_age_days=30):
    """
    Clean up old suspicious images if storage exceeds limit or images are too old.

    ✅ PHASE 2.5: Disk space management to prevent unbounded growth

    Args:
        max_storage_mb: Maximum storage in megabytes (default 1 GB)
        max_age_days: Maximum age of images in days (default 30)
    """
    try:
        storage_dir = Path(CONFIG['STORAGE_DIR'])

        if not storage_dir.exists():
            return

        # Get all image files
        image_files = list(storage_dir.glob('*.jpg'))

        if not image_files:
            return

        # Calculate total size
        total_size = sum(f.stat().st_size for f in image_files)
        total_size_mb = total_size / (1024 * 1024)

        logger.info(f"Storage check: {total_size_mb:.2f} MB used by {len(image_files)} images")

        # Delete old files if over limit
        if total_size_mb > max_storage_mb:
            logger.warning(f"Storage exceeds limit: {total_size_mb:.2f} MB > {max_storage_mb} MB")

            # Sort by modification time (oldest first)
            image_files.sort(key=lambda f: f.stat().st_mtime)

            deleted = 0
            freed_mb = 0

            for img_file in image_files:
                # Keep deleting until we reach 80% of limit
                if total_size_mb - freed_mb <= max_storage_mb * 0.8:
                    break

                try:
                    file_size = img_file.stat().st_size / (1024 * 1024)
                    img_file.unlink()
                    freed_mb += file_size
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete {img_file}: {e}")

            if deleted > 0:
                logger.info(f"Cleanup: Deleted {deleted} old images, freed {freed_mb:.2f} MB")

        # Also delete files older than max_age_days
        current_time = time.time()
        age_threshold = max_age_days * 24 * 60 * 60  # Convert days to seconds

        deleted_old = 0
        for img_file in image_files:
            try:
                if current_time - img_file.stat().st_mtime > age_threshold:
                    img_file.unlink()
                    deleted_old += 1
            except Exception as e:
                logger.error(f"Failed to delete old file {img_file}: {e}")

        if deleted_old > 0:
            logger.info(f"Cleanup: Deleted {deleted_old} images older than {max_age_days} days")

    except Exception as e:
        logger.error(f"Disk cleanup error: {e}")


def save_suspicious_image(image_data, device_id, prediction):
    """Save image if classified as fake."""
    try:
        if not prediction['is_fake']:
            return

        # Create storage directory
        storage_dir = Path(CONFIG['STORAGE_DIR'])
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"fake_{device_id}_{timestamp}.jpg"
        filepath = storage_dir / filename

        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_data)

        logger.info(f"Saved suspicious image: {filename}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return None


def log_prediction(device_id, prediction, image_path=None):
    """Log prediction to file and update statistics (thread-safe)."""
    try:
        # Log to file (outside lock - I/O operations)
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'device_id': device_id,
            'is_fake': prediction['is_fake'],
            'fake_probability': prediction['fake_probability'],
            'confidence': prediction['confidence'],
            'image_path': image_path
        }

        logger.info(json.dumps(log_entry))

        # ✅ PHASE 2.4: Acquire lock for thread-safe device_stats access
        with stats_lock:
            # Update statistics
            if device_id not in device_stats:
                device_stats[device_id] = {
                    'total_images': 0,
                    'fake_detections': 0,
                    'last_prediction': None,
                    'avg_fake_prob': 0.0
                }

            stats = device_stats[device_id]
            stats['total_images'] += 1
            if prediction['is_fake']:
                stats['fake_detections'] += 1
            stats['last_prediction'] = datetime.now().isoformat()

            # Update average fake probability
            avg = stats['avg_fake_prob']
            stats['avg_fake_prob'] = (
                (avg * (stats['total_images'] - 1) + prediction['fake_probability'])
                / stats['total_images']
            )

        # Save stats (also thread-safe)
        save_stats()
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


def save_stats():
    """Save device statistics to JSON file (thread-safe)."""
    try:
        stats_dir = Path(CONFIG['STATS_FILE']).parent
        stats_dir.mkdir(parents=True, exist_ok=True)

        # ✅ PHASE 2.4: Also protect file writes with lock
        with stats_lock:
            with open(CONFIG['STATS_FILE'], 'w') as f:
                json.dump(device_stats, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save stats: {e}")


def send_alerts(device_id, prediction):
    """Send alerts via multiple channels."""
    if not prediction['is_fake']:
        return

    alert_msg = (
        f"🚨 DEEPFAKE DETECTED\n"
        f"Device: {device_id}\n"
        f"Probability: {prediction['fake_probability']:.3f}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Log file alert (always enabled)
    logger.warning(f"ALERT: {alert_msg}")

    # Email alert
    if CONFIG['EMAIL_CONFIG']['enabled']:
        send_email_alert(device_id, prediction, alert_msg)

    # Webhook alert
    if CONFIG['WEBHOOK_URL']:
        send_webhook_alert(device_id, prediction)


def send_email_alert(device_id, prediction, message):
    """Send email alert (background thread)."""
    def _send():
        try:
            config = CONFIG['EMAIL_CONFIG']
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = config['recipient_email']
            msg['Subject'] = f'[ALERT] Deepfake Detected on Device {device_id}'
            msg.attach(MIMEText(message, 'plain'))

            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['sender_email'], config['sender_password'])
                server.send_message(msg)

            logger.info(f"Email alert sent to {config['recipient_email']}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    thread = Thread(target=_send, daemon=True)
    thread.start()


def send_webhook_alert(device_id, prediction):
    """Send webhook alert (background thread)."""
    def _send():
        try:
            payload = {
                'device_id': device_id,
                'timestamp': datetime.now().isoformat(),
                'is_fake': prediction['is_fake'],
                'fake_probability': prediction['fake_probability'],
                'confidence': prediction['confidence']
            }

            response = requests.post(
                CONFIG['WEBHOOK_URL'],
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"Webhook alert sent successfully")
            else:
                logger.warning(f"Webhook returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    thread = Thread(target=_send, daemon=True)
    thread.start()


@app.route('/test', methods=['GET', 'OPTIONS'])
def test():
    """Simple test endpoint to verify server connectivity."""
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200


@app.route('/predict', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")  # Max 10 predictions per minute per IP
def predict():
    """
    Receive image from Nicla device and return prediction.

    Expected request:
    - device_id: Device identifier (string)
    - image: JPEG image data (bytes)

    Returns:
    {
        'device_id': str,
        'is_fake': bool,
        'fake_probability': float (0-1),
        'confidence': float (0-1),
        'inference_time_ms': float
    }
    """
    try:
        # Log request details
        logger.info(f"Received prediction request from {request.remote_addr}")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files: {request.files}")

        # Validate request
        if 'device_id' not in request.form:
            logger.error("Missing device_id in request")
            return jsonify({'error': 'Missing device_id'}), 400

        if 'image' not in request.files:
            logger.error("Missing image in request")
            return jsonify({'error': 'Missing image'}), 400

        device_id = request.form['device_id']

        # ✅ PHASE 2.3: Validate device_id format (prevent path traversal)
        # Only allow alphanumeric characters, underscores, and hyphens
        if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', device_id):
            logger.warning(f"Invalid device_id format: {device_id}")
            return jsonify({'error': 'Invalid device_id format. Only alphanumeric, underscore, hyphen allowed (1-50 chars)'}), 400

        image_file = request.files['image']

        # ✅ PHASE 2.3: Validate image file size (prevent DoS via large files)
        MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB max
        image_file.seek(0, os.SEEK_END)
        file_size = image_file.tell()
        image_file.seek(0)

        if file_size > MAX_IMAGE_SIZE:
            logger.warning(f"Image too large: {file_size} bytes from {device_id}")
            return jsonify({'error': f'Image too large: {file_size} bytes (max 5 MB)'}), 400

        if file_size == 0:
            logger.warning(f"Empty image file from {device_id}")
            return jsonify({'error': 'Image file is empty'}), 400

        image_data = image_file.read()
        logger.info(f"Processing image for device: {device_id}, size: {len(image_data)} bytes")

        # Preprocess image
        image_np = preprocess_image(image_data)
        if image_np is None:
            return jsonify({'error': 'Image preprocessing failed'}), 400

        # Run inference
        import time
        start_time = time.time()
        patch_logits, is_fake, fake_prob, confidence = run_inference(image_np)
        inference_time = (time.time() - start_time) * 1000

        if patch_logits is None:
            return jsonify({'error': 'Inference failed'}), 500

        # Build prediction response (convert numpy types to Python native types for JSON serialization)
        prediction = {
            'device_id': device_id,
            'is_fake': bool(is_fake),  # Convert numpy bool to Python bool
            'fake_probability': float(fake_prob),  # Ensure Python float
            'confidence': float(confidence),  # Ensure Python float
            'inference_time_ms': float(inference_time)  # Ensure Python float
        }

        # Save suspicious image
        image_path = save_suspicious_image(image_data, device_id, prediction)

        # Log prediction
        log_prediction(device_id, prediction, image_path)

        # Send alerts if fake
        send_alerts(device_id, prediction)

        # ✅ PHASE 2.5: Periodic cleanup of old images
        global cleanup_counter
        cleanup_counter += 1
        if cleanup_counter % 100 == 0:  # Cleanup every 100 predictions
            logger.info(f"Running periodic disk cleanup (prediction #{cleanup_counter})")
            cleanup_old_images(max_storage_mb=1000, max_age_days=30)

        return jsonify(prediction), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/capture', methods=['POST', 'OPTIONS'])
@limiter.limit("60 per minute")  # Max 60 captures/minute per IP
def capture():
    """
    Receive image from Nicla Vision device without running inference.
    Used for capturing reference images or for manual review.

    Expected request:
    - device_id: Device identifier (string)
    - image: JPEG image data (bytes)
    - capture_type: Optional classification ('reference', 'suspicious', 'normal')

    Returns:
    {
        'device_id': str,
        'filename': str,
        'timestamp': str,
        'size_bytes': int,
        'capture_type': str
    }
    """
    try:
        # Validate request
        if 'device_id' not in request.form:
            return jsonify({'error': 'Missing device_id'}), 400

        if 'image' not in request.files:
            return jsonify({'error': 'Missing image'}), 400

        device_id = request.form['device_id']
        image_file = request.files['image']
        image_data = image_file.read()
        capture_type = request.form.get('capture_type', 'normal')

        logger.info(f"Received capture request from device: {device_id}, type: {capture_type}, size: {len(image_data)} bytes")

        # Validate image
        try:
            Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Invalid image format: {e}")
            return jsonify({'error': 'Invalid image format'}), 400

        # Create storage directory for captures
        capture_dir = Path('storage/captured_images') / capture_type
        capture_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]  # milliseconds
        filename = f"{capture_type}_{device_id}_{timestamp}.jpg"
        filepath = capture_dir / filename

        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_data)

        logger.info(f"✓ Captured image saved: {filename}")

        # Store metadata
        if device_id not in captured_images:
            captured_images[device_id] = []

        image_metadata = {
            'filename': filename,
            'filepath': str(filepath),
            'timestamp': datetime.now().isoformat(),
            'size_bytes': len(image_data),
            'capture_type': capture_type,
            'device_id': device_id
        }

        captured_images[device_id].append(image_metadata)

        # Keep only last 50 captures per device
        if len(captured_images[device_id]) > 50:
            old_image = captured_images[device_id].pop(0)
            try:
                Path(old_image['filepath']).unlink()
                logger.info(f"Deleted old capture: {old_image['filename']}")
            except:
                pass

        return jsonify(image_metadata), 200

    except Exception as e:
        logger.error(f"Capture error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/captures', methods=['GET'])
@limiter.limit("30 per minute")  # Max 30 requests/minute
def get_captures():
    """Get captured images metadata for all devices."""
    return jsonify(captured_images), 200


@app.route('/api/captures/<device_id>', methods=['GET'])
@limiter.limit("30 per minute")
def get_device_captures(device_id):
    """Get captured images metadata for a specific device."""
    if device_id not in captured_images:
        return jsonify({'device_id': device_id, 'captures': []}), 200

    return jsonify({
        'device_id': device_id,
        'captures': captured_images[device_id]
    }), 200


@app.route('/captured-image/<path:filepath>', methods=['GET'])
@limiter.limit("60 per minute")
def serve_captured_image(filepath):
    """Serve captured image file."""
    try:
        file_path = Path('storage/captured_images') / filepath
        if not file_path.exists():
            return jsonify({'error': 'Image not found'}), 404

        return send_from_directory(file_path.parent, file_path.name)
    except Exception as e:
        logger.error(f"Error serving captured image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
@limiter.limit("30 per minute")  # Max 30 requests/minute
def get_stats():
    """Return statistics for all devices."""
    return jsonify(device_stats), 200


@app.route('/status', methods=['GET'])
@limiter.limit("30 per minute")  # Max 30 requests/minute
def get_status():
    """Return server status."""
    return jsonify({
        'status': 'running',
        'model_loaded': onnx_session is not None,
        'threshold': CONFIG['THRESHOLD'],
        'devices': list(device_stats.keys())
    }), 200


@app.route('/', methods=['GET'])
def serve_dashboard():
    """Serve the web dashboard."""
    return send_from_directory('.', 'dashboard.html')


@app.route('/static/<path:path>', methods=['GET'])
def serve_static(path):
    """Serve static assets."""
    return send_from_directory('static', path)


@app.route('/api/dashboard-data', methods=['GET'])
@limiter.limit("60 per minute")  # Max 60 requests/minute (allow rapid polling)
def get_dashboard_data():
    """Provide data for dashboard updates."""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'stats': device_stats,
        'alerts_enabled': {
            'email': CONFIG['EMAIL_CONFIG']['enabled'],
            'webhook': CONFIG['WEBHOOK_URL'] is not None
        }
    }), 200


if __name__ == '__main__':
    # Initialize model
    if not initialize_model():
        print("Failed to load model. Exiting.")
        exit(1)

    # Create necessary directories
    Path(CONFIG['STORAGE_DIR']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['STATS_FILE']).parent.mkdir(parents=True, exist_ok=True)

    # Run Flask server
    print("\n" + "="*80)
    print("DEEPFAKE DETECTION - RASPBERRY PI SERVER")
    print("="*80)
    print(f"✓ Model loaded: {CONFIG['MODEL_PATH']}")
    print(f"✓ Storage directory: {CONFIG['STORAGE_DIR']}")
    print(f"✓ Detection threshold: {CONFIG['THRESHOLD']}")
    print(f"\nServer running on http://0.0.0.0:5000")
    print("="*80 + "\n")

    # Run on all interfaces (0.0.0.0) so Nicla devices can reach it
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
