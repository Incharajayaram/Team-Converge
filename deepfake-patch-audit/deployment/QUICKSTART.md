# Quick Start Guide - 5 Minutes to Deployment

## Step 1: Export Model to ONNX (2 minutes)

Run this command in the project root directory:

```bash
cd /home/incharaj/Team-Converge/deepfake-patch-audit

python3 scripts/export_student.py \
    --model outputs/checkpoints_two_stage/student_final.pt \
    --onnx-only
```

**Expected output:**
```
================================================================================
DEEPFAKE DETECTION MODEL - DEPLOYMENT PIPELINE
================================================================================
Model: outputs/checkpoints_two_stage/student_final.pt
Output directory: deployment/pi_server/models
Quantization: none
Device: cuda
ONNX Only: True

LOADING TRAINED MODEL
✓ Model loaded from outputs/checkpoints_two_stage/student_final.pt
  Parameters: 1297

PYTORCH TO ONNX EXPORT
...
✓ Model exported to deployment/pi_server/models/deepfake_detector.onnx
✓ ONNX model validation passed

DEPLOYMENT SUMMARY
✓ DEPLOYMENT PIPELINE COMPLETE
```

**Result:** Model file created at `deployment/pi_server/models/deepfake_detector.onnx` (3-5 MB)

## Step 2: Verify Setup (1 minute)

```bash
python3 deployment/test_flask_server.py
```

**Expected output:**
```
TEST SUMMARY
================================================================================
✓ PASSED: Model Loading
✓ PASSED: Image Preprocessing
✓ PASSED: End-to-End Inference
✓ PASSED: Flask App Structure

Total: 4/4 tests passed

✓ All tests passed! Flask server is ready for deployment.
```

## Step 3: Deploy to Raspberry Pi (2 minutes)

### 3.1 Copy Files to Raspberry Pi

```bash
# From your development machine:
scp -r deployment/ pi@<YOUR_PI_IP>:~/deepfake-detection/
scp requirements.txt pi@<YOUR_PI_IP>:~/deepfake-detection/

# Or if you prefer rsync:
rsync -avz deployment/ pi@<YOUR_PI_IP>:~/deepfake-detection/
```

### 3.2 Install Dependencies on Raspberry Pi

```bash
# SSH into Raspberry Pi
ssh pi@<YOUR_PI_IP>

# Install dependencies
cd ~/deepfake-detection
pip3 install -r requirements.txt

# For Raspberry Pi specifically (optional, for better performance):
pip3 install onnxruntime-rpi
```

### 3.3 Start Flask Server

```bash
cd ~/deepfake-detection
python3 pi_server/app.py
```

**Expected output:**
```
================================================================================
DEEPFAKE DETECTION - RASPBERRY PI SERVER
================================================================================
✓ Model loaded: models/deepfake_detector.onnx
✓ Storage directory: storage/suspicious_images
✓ Detection threshold: 0.84

Server running on http://0.0.0.0:5000
================================================================================
```

## Step 4: Access Dashboard

Open your browser and navigate to:
```
http://<YOUR_PI_IP>:5000
```

You should see the dashboard with:
- Real-time device status
- Detection metrics
- Recent alerts
- Suspicious image gallery

## Step 5: Configure Nicla Devices (Optional)

Edit `deployment/nicla/deepfake_detector.ino`:

```cpp
// Line 21: WiFi SSID
const char* WIFI_SSID = "YOUR_SSID";

// Line 22: WiFi Password
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

// Line 28: Raspberry Pi IP Address
const char* PI_SERVER_IP = "192.168.1.100";  // Change to your Pi's IP

// Line 33: Device ID (change for each Nicla)
const String DEVICE_ID = "nicla_1";  // nicla_1, nicla_2, etc.
```

Then upload using Arduino IDE or Arduino CLI.

## Verify Everything Works

### Test 1: Check Server Status

```bash
curl http://<PI_IP>:5000/status

# Expected response:
{
  "status": "running",
  "model_loaded": true,
  "threshold": 0.84,
  "devices": []
}
```

### Test 2: Send Test Image

```bash
# Create a test image or use existing one
curl -X POST http://<PI_IP>:5000/predict \
  -F "device_id=test_device" \
  -F "image=@test_image.jpg"

# Expected response:
{
  "device_id": "test_device",
  "is_fake": false,
  "fake_probability": 0.234,
  "confidence": 0.766,
  "inference_time_ms": 85.5
}
```

### Test 3: Check Logs

```bash
# On Raspberry Pi
tail -f deepfake_detections.log

# Should show:
2024-01-15 10:30:45,123 - INFO - {'timestamp': ..., 'device_id': 'test_device', ...}
```

## Troubleshooting

### Model not found after export

```bash
# Verify file exists:
ls -lh deployment/pi_server/models/deepfake_detector.onnx

# If missing, run export again:
python3 scripts/export_student.py --model outputs/checkpoints_two_stage/student_final.pt --onnx-only
```

### Can't access Flask server on Pi

```bash
# 1. Check Flask is running:
ssh pi@<PI_IP>
ps aux | grep "python.*app.py"

# 2. Check firewall:
sudo ufw allow 5000/tcp

# 3. Check Pi IP:
hostname -I
```

### Nicla can't reach Pi

```bash
# 1. Verify both on same WiFi:
# Check on Pi:
iwconfig  # or: ip addr | grep inet

# 2. Verify IP in Nicla sketch is correct:
# Edit deployment/nicla/deepfake_detector.ino line 28

# 3. Test ping from Pi to Nicla:
# Once Nicla connects to WiFi, note its IP
ping <NICLA_IP>
```

## File Structure

```
deployment/
├── README.md                      # Complete documentation
├── QUICKSTART.md                  # This file
├── requirements.txt               # Python dependencies
├── export_model.py               # Model export script (alternative)
├── test_flask_server.py          # Test suite
├── pi_server/
│   ├── app.py                    # Flask server (main)
│   ├── dashboard.html            # Web UI
│   └── models/
│       └── deepfake_detector.onnx # Exported ONNX model (created by export script)
├── nicla/
│   └── deepfake_detector.ino     # Arduino sketch for Nicla Vision
└── storage/                       # Auto-created on first run
    ├── suspicious_images/        # Fake detections saved here
    └── stats.json               # Device statistics
```

## Expected Performance

| Component | Time/Size |
|-----------|-----------|
| Model size | 3-5 MB (ONNX) |
| Inference | 50-100 ms per image |
| Dashboard update | 5 seconds |
| Throughput | 3-5 images/second |
| End-to-end latency | 200-300 ms |

## Configuration

Key configuration in `pi_server/app.py`:

```python
CONFIG = {
    'MODEL_PATH': 'models/deepfake_detector.onnx',     # Model location
    'STORAGE_DIR': 'storage/suspicious_images',        # Where to save fakes
    'THRESHOLD': 0.84,                                 # Detection threshold
    'EMAIL_CONFIG': {...},                             # Optional email alerts
    'WEBHOOK_URL': None,                               # Optional webhook alerts
}
```

## What's Next?

1. **Monitor the dashboard** at http://<PI_IP>:5000
2. **Check logs** for any errors: `tail -f deepfake_detections.log`
3. **Configure alerts** (email/webhook) in app.py if needed
4. **Add more Nicla devices** by uploading sketch to additional boards
5. **Test under load** with multiple Niclas simultaneously

## Support

- Check logs: `tail -f deepfake_detections.log`
- Check server status: `curl http://PI_IP:5000/status`
- View stats: `curl http://PI_IP:5000/stats`
- Dashboard data: `curl http://PI_IP:5000/api/dashboard-data`

---

That's it! You now have a complete edge-hub deepfake detection system running.
