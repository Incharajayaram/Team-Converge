# Deepfake Detection - Edge-Hub Architecture

Complete federated learning deployment for deepfake detection using Nicla Vision devices (edge) and Raspberry Pi (hub).

## Architecture Overview

```
Nicla Vision Devices (4x)        Raspberry Pi Hub (1x)
─────────────────────────        ──────────────────────
• Camera capture (2MP)           • ONNX model inference
• Resize to 128x128              • Flask REST API server
• JPEG compression (80%)         • Web dashboard UI
• WiFi HTTP POST                 • Alert system
• LED feedback (green/red)        • Image storage
```

## Quick Start

### 1. Export Model to ONNX Format

On your development machine:

```bash
cd /home/incharaj/Team-Converge/deepfake-patch-audit

# Export to ONNX only (recommended for Flask server)
python scripts/export_student.py \
    --model outputs/checkpoints_two_stage/student_final.pt \
    --onnx-only

# Output: deployment/pi_server/models/deepfake_detector.onnx
```

### 2. Set Up Raspberry Pi

#### 2.1 Install Dependencies

```bash
# SSH into Raspberry Pi
ssh pi@<PI_IP>

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python dependencies
pip install -r requirements.txt

# Specific for ARM (Raspberry Pi)
pip install onnxruntime-rpi
```

#### 2.2 Copy Deployment Files

From your development machine:

```bash
# Copy entire deployment directory to Pi
scp -r deployment/ pi@<PI_IP>:~/deepfake-detection/

# Or copy just the Flask server files
scp -r deployment/pi_server/ pi@<PI_IP>:~/deepfake-detection/
scp deployment/requirements.txt pi@<PI_IP>:~/deepfake-detection/
```

#### 2.3 Verify Model Location

On Raspberry Pi:

```bash
ls -lh ~/deepfake-detection/pi_server/models/deepfake_detector.onnx
# Should show: deepfake_detector.onnx (3-5 MB)
```

### 3. Start Flask Server on Raspberry Pi

```bash
cd ~/deepfake-detection/

# Run the Flask server (will listen on 0.0.0.0:5000)
python pi_server/app.py

# Output should show:
# ✓ Model loaded: models/deepfake_detector.onnx
# ✓ Storage directory: storage/suspicious_images
# ✓ Detection threshold: 0.84
# Server running on http://0.0.0.0:5000
```

### 4. Access Web Dashboard

Open browser and navigate to:
```
http://<PI_IP>:5000
```

You should see:
- Real-time monitoring of connected Nicla devices
- Detection statistics and metrics
- Alert log
- Suspicious image gallery

## Nicla Vision Configuration

### 4.1 Prerequisites

- Arduino IDE or Arduino CLI
- Nicla Vision board
- USB cable for programming
- WiFi credentials

### 4.2 Configure Arduino Sketch

Edit `deployment/nicla/deepfake_detector.ino`:

```cpp
// Line 21-25: WiFi Configuration
const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

// Line 28-30: Raspberry Pi Configuration
const char* PI_SERVER_IP = "192.168.1.100";  // Your Pi's IP
const int PI_SERVER_PORT = 5000;

// Line 33: Device Identification (change for each Nicla)
const String DEVICE_ID = "nicla_1";  // nicla_1, nicla_2, nicla_3, nicla_4
```

### 4.3 Upload Sketch

```bash
# Using Arduino IDE:
# 1. Open deployment/nicla/deepfake_detector.ino
# 2. Select Board: Arduino Nicla Vision
# 3. Select Port: /dev/ttyACM0 (or COM port on Windows)
# 4. Click Upload

# Or using Arduino CLI:
arduino-cli compile --fqbn arduino:mbed_portenta:nicla_vision deployment/nicla/deepfake_detector.ino
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:mbed_portenta:nicla_vision deployment/nicla/
```

### 4.4 Verify Connection

After uploading sketch:

1. Open Serial Monitor (Arduino IDE)
2. Set baud rate to 115200
3. You should see output:
```
=== Deepfake Detection Client ===
Device ID: nicla_1
Initializing camera...
Connecting to WiFi...
```

4. Check Flask server logs for device connections:
```
2024-01-15 10:30:45,123 - INFO - Device nicla_1 connected
```

## API Endpoints

### `/predict` (POST)

Receive image from Nicla device and return prediction.

**Request:**
```http
POST /predict HTTP/1.1
Host: <PI_IP>:5000
Content-Type: multipart/form-data

form-data:
  device_id: "nicla_1"
  image: <JPEG binary data>
```

**Response (200 OK):**
```json
{
  "device_id": "nicla_1",
  "is_fake": false,
  "fake_probability": 0.234,
  "confidence": 0.766,
  "inference_time_ms": 85.5
}
```

### `/stats` (GET)

Get statistics for all devices.

**Response:**
```json
{
  "nicla_1": {
    "total_images": 125,
    "fake_detections": 3,
    "last_prediction": "2024-01-15T10:30:45.123456",
    "avg_fake_prob": 0.45
  }
}
```

### `/status` (GET)

Get server status.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "threshold": 0.84,
  "devices": ["nicla_1", "nicla_2"]
}
```

### `/api/dashboard-data` (GET)

Get data for web dashboard.

## Configuration

Edit `pi_server/app.py` to customize:

```python
CONFIG = {
    'MODEL_PATH': 'models/deepfake_detector.onnx',
    'STORAGE_DIR': 'storage/suspicious_images',
    'STATS_FILE': 'storage/stats.json',
    'THRESHOLD': 0.84,  # Calibrated detection threshold
    'EMAIL_CONFIG': {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': 'your-email@gmail.com',
        'sender_password': 'app-password',
        'recipient_email': 'alert@example.com'
    },
    'WEBHOOK_URL': 'https://your-webhook.com/alerts'
}
```

## Alert System

The system supports multiple alert channels:

### 1. Dashboard Notifications
- Real-time updates in web UI
- No additional configuration needed

### 2. Log File
- Stored in `deepfake_detections.log`
- Contains all predictions and alerts
- Automatically created on first run

### 3. Email Alerts (Optional)
- Enable in `CONFIG['EMAIL_CONFIG']`
- Requires SMTP credentials
- Sends email when deepfake detected

### 4. Webhook Alerts (Optional)
- Set `CONFIG['WEBHOOK_URL']`
- Sends POST with prediction data
- Useful for integration with other systems

## Performance Metrics

### Expected Performance

| Metric | Value |
|--------|-------|
| Model Input | 256x256 RGB image |
| Model Output | (1, 1, 126, 126) patch-logits |
| Inference Time | 50-100 ms (Raspberry Pi 4) |
| Memory Usage | ~100-150 MB |
| Model Size | 3-5 MB (ONNX) |
| Bandwidth per Image | 10-15 KB (128x128 JPEG @ 80%) |
| End-to-End Latency | 200-300 ms |
| Max Throughput | 3-5 images/sec (all 4 Niclas) |

### Detection Threshold

- **Threshold**: 0.84 (calibrated)
- **P(fake) > 0.84** → Classified as DEEPFAKE
- **P(fake) ≤ 0.84** → Classified as REAL

## Troubleshooting

### 1. "Model not found" error

```bash
# Check model file exists
ls -l deployment/pi_server/models/deepfake_detector.onnx

# If missing, re-export:
python scripts/export_student.py --model outputs/checkpoints_two_stage/student_final.pt --onnx-only
```

### 2. Nicla can't connect to Raspberry Pi

```bash
# 1. Verify Pi IP is correct in Arduino sketch
# 2. Check Nicla and Pi are on same WiFi network
# 3. Test connection from Nicla:
ping <PI_IP>

# 4. Check Flask server is running on Pi:
ps aux | grep "python.*app.py"

# 5. Check firewall (on Pi):
sudo ufw status
sudo ufw allow 5000/tcp  # If needed
```

### 3. Slow inference on Raspberry Pi

```bash
# Check system resources:
free -h  # Available memory
top     # CPU usage

# Optimize:
# - Reduce other services
# - Use ONNX quantization (if needed)
# - Reduce image capture frequency in Nicla sketch
```

### 4. Storage space issues

```bash
# Check disk space:
df -h

# Clean old suspicious images:
rm storage/suspicious_images/fake_*.jpg

# Or change storage policy in app.py
```

## Testing

### Test Single Image

```bash
curl -X POST http://PI_IP:5000/predict \
  -F "device_id=test_device" \
  -F "image=@test_image.jpg"
```

### Load Test (Multiple Niclas)

```bash
# From Nicla, set capture interval
const int CAPTURE_INTERVAL_MS = 1000;  // 1 second

# Run for 5-10 minutes and monitor:
# - Dashboard metrics
# - Server logs
# - Memory usage
```

## Deployment Checklist

- [ ] Model exported to ONNX
- [ ] Flask dependencies installed on Pi
- [ ] Deployment files copied to Pi
- [ ] Flask server running and accessible
- [ ] Dashboard accessible at http://<PI_IP>:5000
- [ ] Nicla sketch configured with correct IP
- [ ] Nicla can reach Pi (ping test)
- [ ] First image captured and prediction received
- [ ] LED feedback working (green/red)
- [ ] Alert system tested
- [ ] All 4 Niclas configured and connected

## Advanced Configuration

### Custom Model Path

```bash
# Export to custom location
python scripts/export_student.py \
    --model outputs/checkpoints_two_stage/student_final.pt \
    --output-dir /path/to/deployment/models \
    --onnx-only
```

### Enable Quantization (Smaller Model)

```bash
# Export with int8 quantization (3-5 MB → 1-2 MB)
python scripts/export_student.py \
    --model outputs/checkpoints_two_stage/student_final.pt \
    --quantization dynamic \
    --output-dir deployment/pi_server/models
```

### Webhook Integration Example

```python
# In app.py CONFIG:
'WEBHOOK_URL': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

# Will POST to Slack when deepfake detected
```

## Logs and Storage

- **Predictions Log**: `deepfake_detections.log`
- **Statistics**: `storage/stats.json`
- **Suspicious Images**: `storage/suspicious_images/fake_*.jpg`
- **Dashboard Data**: Updated in real-time from `/api/dashboard-data`

## Support

For issues or questions:
1. Check logs: `tail -f deepfake_detections.log`
2. Check server status: `curl http://PI_IP:5000/status`
3. Verify network: `ping PI_IP` from Nicla
4. Check model: `ls -l models/deepfake_detector.onnx`

---

**Architecture by**: Team Converge
**Model**: TinyLaDeDa (1,297 parameters)
**Deployment Framework**: Flask + ONNX Runtime
**Target Hardware**: Raspberry Pi 4 (4GB+), Nicla Vision (4x)
