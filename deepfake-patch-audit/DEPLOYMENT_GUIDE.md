# Complete Deployment & Enhancement Guide

This guide covers the full implementation plan to take the deepfake detection system from 85% to 100% completion, add production robustness, and implement the remote capture feature.

**Last Updated**: January 2026
**Status**: Planning Phase Complete → Implementation Starting

---

## Table of Contents

1. [Phase 1: Complete Original Architecture](#phase-1-complete-original-architecture)
2. [Phase 2: Backend Robustness](#phase-2-backend-robustness)
3. [Phase 3: Remote Capture Feature](#phase-3-remote-capture-feature)
4. [Deployment Checklist](#deployment-checklist)
5. [Testing Procedures](#testing-procedures)

---

## Phase 1: Complete Original Architecture

**Timeline**: 10-15 hours
**Status**: NOT STARTED

This phase completes the core system that was designed but not fully implemented.

### Step 1.1: Export ONNX Model

**Time**: 5 minutes
**Priority**: 🔴 CRITICAL - BLOCKER

**Current Issue**:
- Model file missing at `deployment/pi_server/models/deepfake_detector.onnx`
- System cannot run inference without this

**Implementation**:

```bash
# Navigate to project root
cd /home/incharaj/Team-Converge/deepfake-patch-audit

# Export model to ONNX format
python scripts/export_student.py \
    --model outputs/checkpoints_two_stage/student_final.pt \
    --onnx-only

# Verify the file was created
ls -lh deployment/pi_server/models/deepfake_detector.onnx
# Expected output: ~3-5 MB file
```

**Validation**:

```bash
# Test model loading
cd deployment/pi_server
python -c "
import onnxruntime as rt
session = rt.InferenceSession('models/deepfake_detector.onnx')
inputs = session.get_inputs()
print(f'Model loaded successfully!')
print(f'Input shape: {inputs[0].shape}')
print(f'Input type: {inputs[0].type}')
"
```

**Expected Output**:
```
Model loaded successfully!
Input shape: [1, 3, 256, 256]
Input type: float32
```

**✅ Success Criteria**:
- Model file exists and is 3-5 MB
- Model loads without errors
- Input shape is [1, 3, 256, 256]

---

### Step 1.2: Implement Nicla Image Capture

**Time**: 4-6 hours
**Priority**: 🔴 CRITICAL - BLOCKER
**Complexity**: High (hardware-dependent)

**File**: `deployment/nicla/deepfake_detector.ino`
**Lines**: 210-243 (currently returns `nullptr`)

**Current Code** (PLACEHOLDER):
```cpp
uint8_t* captureImage(int* imageSize) {
    // TODO: Implement camera capture
    *imageSize = 0;
    return nullptr;
}
```

**What It Should Do**:
1. Initialize camera (2MP OV7670)
2. Capture frame from camera
3. Return pointer to frame data
4. Set imageSize to frame length

**Implementation Approach**:

```cpp
#include <camera.h>

bool initializeCamera() {
    /*
    Initialize camera with 320x240 resolution (2MP reduced).
    This must be called once during setup.
    */
    if (!Camera.begin(CAMERA_R320x240, PIXFORMAT_RGB565, 1)) {
        Serial.println("❌ Camera initialization failed!");
        ledBlink(LED_RED, 3);  // Blink red 3 times
        return false;
    }

    // Configure camera parameters
    Camera.setBrightness(0);    // Default brightness
    Camera.setContrast(0);      // Default contrast
    Camera.setSaturation(0);    // Default saturation

    Serial.println("✅ Camera initialized: 320x240 RGB565");
    return true;
}

uint8_t* captureImage(int* imageSize) {
    /*
    Capture a single frame from the camera.

    Returns:
        - Pointer to frame buffer on success
        - nullptr on failure

    Sets imageSize to the frame length in bytes.
    */

    // Attempt to grab frame
    camera_fb_t *frame = Camera.grab();

    if (!frame) {
        Serial.println("❌ Camera capture failed!");
        *imageSize = 0;
        return nullptr;
    }

    // Return frame data
    *imageSize = frame->len;

    Serial.print("✅ Captured frame: ");
    Serial.print(frame->width);
    Serial.print("x");
    Serial.print(frame->height);
    Serial.print(" (");
    Serial.print(*imageSize);
    Serial.println(" bytes)");

    return frame->buf;
}
```

**Key Points**:
- Frame is 320x240 RGB565 format
- Need to resize to 128x128 in next step (jpegCompress)
- Camera returns pointer to internal buffer
- Must use frame immediately before next capture

**Hardware Requirements**:
- Arduino Nicla Vision board
- OV7670 camera sensor (built-in)
- 2MP capability

**Testing Before Moving Forward**:

```cpp
// Add this to loop() temporarily to test
void testCameraCapture() {
    static unsigned long lastTest = 0;

    if (millis() - lastTest > 5000) {  // Every 5 seconds
        lastTest = millis();

        int size = 0;
        uint8_t* frame = captureImage(&size);

        if (frame && size > 0) {
            Serial.print("✅ Test passed: ");
            Serial.println(size);
            ledOn(LED_GREEN, 1000);  // Green for 1 second
        } else {
            Serial.println("❌ Test failed");
            ledBlink(LED_RED, 1);
        }
    }
}
```

**Troubleshooting**:

| Problem | Solution |
|---------|----------|
| "Camera init failed" | Check I2C connection, power supply |
| Frame size is 0 | Wait longer after initialization |
| Image looks wrong | Check brightness/contrast settings |
| Memory errors | Reduce resolution or frame buffer |

**✅ Success Criteria**:
- Camera initializes on boot
- Frame capture returns non-null pointer
- Frame size is approximately 153,600 bytes (320×240×2)
- No memory leaks during repeated captures
- Serial output shows "✅" messages

---

### Step 1.3: Implement JPEG Compression

**Time**: 2-3 hours
**Priority**: 🔴 CRITICAL - BLOCKER
**Complexity**: High (library integration)

**File**: `deployment/nicla/deepfake_detector.ino`
**Lines**: 245-269

**Current Code** (PLACEHOLDER):
```cpp
uint8_t* jpegCompress(uint8_t* rawImage, int rawSize, int* compressedSize) {
    // TODO: Implement JPEG compression
    *compressedSize = 0;
    return nullptr;
}
```

**What It Should Do**:
1. Take 320x240 RGB565 image
2. Resize to 128x128 (4x downsampling)
3. Compress to JPEG at 80% quality
4. Return ~10-15 KB compressed image

**Implementation Approach**:

```cpp
#include <JPEGENC.h>

// Global JPEG encoder instance
JPEGENC jpeg;
uint8_t jpegBuffer[20000];  // Buffer for compressed JPEG (max ~15 KB)

uint8_t* jpegCompress(uint8_t* rawImage, int rawSize, int* compressedSize) {
    /*
    Compress RGB565 image to JPEG format.

    Input:
        - rawImage: Pointer to 320x240 RGB565 frame
        - rawSize: Size of raw image data

    Output:
        - Returns pointer to JPEG buffer
        - Sets compressedSize to JPEG size in bytes

    Expected:
        - Input: ~153 KB (320×240×2)
        - Output: ~10-15 KB (80% quality)
        - Time: <100ms
    */

    unsigned long startTime = millis();

    // Initialize JPEG encoder
    if (!jpeg.open(jpegBuffer, sizeof(jpegBuffer))) {
        Serial.println("❌ JPEG encoder open failed");
        *compressedSize = 0;
        return nullptr;
    }

    // Configure encoder: 128x128 resolution, 80% quality
    // Input: 320x240 RGB565
    // Output: 128x128 JPEG
    if (!jpeg.encodeBegin(128, 128, JPEGENC_PIXEL_RGB565,
                         JPEGENC_SUBSAMPLE_420, JPEGENC_Q_HIGH)) {
        Serial.println("❌ JPEG encoding begin failed");
        *compressedSize = 0;
        return nullptr;
    }

    // Add frame to encoder (handles resizing automatically)
    // The encoder will:
    // 1. Resize from 320x240 to 128x128
    // 2. Convert RGB565 to YCbCr (color space for JPEG)
    // 3. Apply JPEG compression
    if (!jpeg.addFrame(rawImage, rawSize)) {
        Serial.println("❌ JPEG frame addition failed");
        *compressedSize = 0;
        return nullptr;
    }

    // Close encoder and get final size
    int compressedBytes = jpeg.close();

    if (compressedBytes <= 0) {
        Serial.println("❌ JPEG encoding failed");
        *compressedSize = 0;
        return nullptr;
    }

    unsigned long endTime = millis();
    unsigned long duration = endTime - startTime;

    *compressedSize = compressedBytes;

    // Log compression statistics
    Serial.print("✅ JPEG compressed: ");
    Serial.print(rawSize);
    Serial.print(" → ");
    Serial.print(compressedBytes);
    Serial.print(" bytes (");
    Serial.print((100.0 * compressedBytes) / rawSize, 1);
    Serial.print("%) in ");
    Serial.print(duration);
    Serial.println(" ms");

    // Validate compression ratio
    if (compressedBytes < 5000 || compressedBytes > 20000) {
        Serial.print("⚠️  Unusual compression: ");
        Serial.print(compressedBytes);
        Serial.println(" bytes");
    }

    return jpegBuffer;
}
```

**JPEGENC Library Setup**:

1. **Install Library** (Arduino IDE):
   - Sketch → Include Library → Manage Libraries
   - Search: "JPEGENC"
   - Install by bitbank2

2. **Alternative** (Manual):
   ```
   Download: https://github.com/bitbank2/JPEGENC
   Place in: ~/Arduino/libraries/JPEGENC/
   ```

**Expected Compression Results**:

| Metric | Value |
|--------|-------|
| Input size | ~153 KB (320×240×2) |
| Output size | 10-15 KB |
| Compression ratio | 85-90% |
| Quality | 80% (good balance) |
| Processing time | 50-100 ms |

**Testing**:

```cpp
// Add to loop() to test compression
void testJpegCompression() {
    static unsigned long lastTest = 0;

    if (millis() - lastTest > 10000) {  // Every 10 seconds
        lastTest = millis();

        int rawSize = 0;
        uint8_t* rawImage = captureImage(&rawSize);

        if (rawImage && rawSize > 0) {
            int jpegSize = 0;
            uint8_t* jpegData = jpegCompress(rawImage, rawSize, &jpegSize);

            if (jpegData && jpegSize > 0) {
                Serial.print("✅ Compression test passed: ");
                Serial.print(jpegSize);
                Serial.println(" bytes");
                ledOn(LED_GREEN, 1000);
            } else {
                Serial.println("❌ Compression failed");
                ledBlink(LED_RED, 1);
            }
        }
    }
}
```

**Troubleshooting**:

| Problem | Solution |
|---------|----------|
| "JPEG encoder open failed" | Increase jpegBuffer size or check memory |
| Compressed size too large (>20KB) | Reduce quality setting, check input resolution |
| Compressed size too small (<5KB) | Input may be corrupted, check capture step |
| Encoding takes >200ms | Normal for Nicla, check if blocking loop |

**✅ Success Criteria**:
- JPEG compression produces 10-15 KB output
- Processing time < 200ms per image
- Compression ratio 85-90%
- Serial output shows "✅ JPEG compressed" messages
- Multiple compressions work without memory leaks

---

### Step 1.4: Fix HTTP Multipart Upload

**Time**: 2-3 hours
**Priority**: 🔴 CRITICAL - BLOCKER
**Complexity**: High (protocol-specific)

**File**: `deployment/nicla/deepfake_detector.ino`
**Lines**: 390-401

**Current Code** (BROKEN):
```cpp
bool sendToPi(uint8_t* imageData, int imageSize) {
    // ... connection code ...

    client.println("POST /predict HTTP/1.1");
    client.println("Host: " + String(PI_SERVER_IP));
    client.println("Content-Type: multipart/form-data; boundary=----Boundary");
    client.println("Connection: close");
    client.println();

    // ❌ MISSING: Actual multipart form-data encoding!

    return true;
}
```

**Problem**:
- Headers are set but no body is sent
- Flask receives empty request
- Device ID and image data never reach server

**What It Should Do**:
1. Build proper multipart/form-data body
2. Include device_id as form field
3. Include JPEG image as binary file field
4. Send with correct Content-Length
5. Wait for and parse response

**Implementation Approach**:

```cpp
bool sendToPi(uint8_t* imageData, int imageSize) {
    /*
    Send JPEG image to Raspberry Pi Flask server via HTTP POST.

    Multipart form-data format:
    --boundary
    Content-Disposition: form-data; name="device_id"

    nicla_1
    --boundary
    Content-Disposition: form-data; name="image"; filename="image.jpg"
    Content-Type: image/jpeg

    <binary JPEG data>
    --boundary--

    Returns:
        - true: Server returned 200 OK or valid JSON response
        - false: Connection failed, timeout, or error response
    */

    // Step 1: Establish TCP connection
    Serial.print("📡 Connecting to ");
    Serial.print(PI_SERVER_IP);
    Serial.print(":");
    Serial.println(PI_SERVER_PORT);

    if (!client.connect(PI_SERVER_IP, PI_SERVER_PORT)) {
        Serial.println("❌ Connection failed!");
        return false;
    }

    Serial.println("✅ Connected to Pi");

    // Step 2: Build multipart form-data body
    String boundary = "----NiclaBoundary12345";

    // Part 1: device_id field
    String bodyStart = "";
    bodyStart += "--" + boundary + "\r\n";
    bodyStart += "Content-Disposition: form-data; name=\"device_id\"\r\n";
    bodyStart += "\r\n";
    bodyStart += DEVICE_ID + "\r\n";

    // Part 2: image file field
    bodyStart += "--" + boundary + "\r\n";
    bodyStart += "Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n";
    bodyStart += "Content-Type: image/jpeg\r\n";
    bodyStart += "\r\n";

    // Part 3: closing boundary
    String bodyEnd = "\r\n--" + boundary + "--\r\n";

    // Step 3: Calculate total content length
    int contentLength = bodyStart.length() + imageSize + bodyEnd.length();

    Serial.print("📦 Payload size: ");
    Serial.print(bodyStart.length());
    Serial.print(" + ");
    Serial.print(imageSize);
    Serial.print(" + ");
    Serial.print(bodyEnd.length());
    Serial.print(" = ");
    Serial.print(contentLength);
    Serial.println(" bytes");

    // Step 4: Send HTTP headers
    client.println("POST /predict HTTP/1.1");
    client.print("Host: ");
    client.println(PI_SERVER_IP);
    client.print("Content-Type: multipart/form-data; boundary=");
    client.println(boundary);
    client.print("Content-Length: ");
    client.println(contentLength);
    client.println("Connection: close");
    client.println();  // Empty line marks end of headers

    // Step 5: Send body start (text field)
    client.print(bodyStart);

    // Step 6: Send binary image data
    client.write(imageData, imageSize);

    // Step 7: Send body end (closing boundary)
    client.print(bodyEnd);

    Serial.println("📤 Request sent, waiting for response...");

    // Step 8: Wait for server response with timeout
    unsigned long timeout = millis();
    const unsigned long RESPONSE_TIMEOUT = 10000;  // 10 second timeout

    while (client.connected() && !client.available()) {
        if (millis() - timeout > RESPONSE_TIMEOUT) {
            Serial.println("❌ Response timeout (>10s)");
            client.stop();
            return false;
        }
        delay(10);
    }

    // Step 9: Read entire response
    String response = "";
    int responseSize = 0;

    while (client.available()) {
        char c = client.read();
        response += c;
        responseSize++;

        // Prevent response from being too large
        if (responseSize > 2000) {
            Serial.println("⚠️  Response truncated (>2KB)");
            break;
        }
    }

    client.stop();

    Serial.println("------- SERVER RESPONSE -------");
    Serial.println(response);
    Serial.println("--------- END RESPONSE --------");

    // Step 10: Parse response
    bool success = false;

    // Check for successful HTTP status
    if (response.indexOf("200 OK") > 0) {
        Serial.println("✅ HTTP 200 OK");
        success = true;
    } else if (response.indexOf("200") > 0) {
        Serial.println("✅ HTTP 200 (implicit)");
        success = true;
    }

    // Also check for valid JSON response (indicates successful processing)
    if (response.indexOf("\"is_fake\"") > 0) {
        Serial.println("✅ Valid prediction response detected");
        success = true;

        // Optional: Extract prediction result
        if (response.indexOf("true") > response.indexOf("\"is_fake\"")) {
            Serial.println("🚨 SERVER DETECTED FAKE!");
            ledBlink(LED_RED, 3);
        } else {
            Serial.println("✅ Server detected REAL image");
            ledOn(LED_GREEN, 1000);
        }
    }

    if (!success) {
        Serial.println("❌ Invalid response from server");
        return false;
    }

    return true;
}
```

**Key Implementation Details**:

1. **Multipart Boundary**:
   - Unique string separating form parts
   - Used in both headers and body
   - Must not appear in image data (unlikely with binary JPEG)

2. **Content-Length**:
   - Must be exact: header_size + image_size + closing_size
   - Server uses this to know when request is complete

3. **Binary Data Handling**:
   - Use `client.write()` for binary data (not println)
   - `client.println()` adds CRLF which corrupts binary data

4. **Response Parsing**:
   - Check for "200 OK" in HTTP status line
   - Also check for JSON response (indicates processing)
   - Extract is_fake from JSON if present

**Testing Checklist**:

```cpp
void testFullPipeline() {
    Serial.println("\n=== FULL PIPELINE TEST ===");

    // Step 1: Capture
    Serial.println("Step 1: Capturing image...");
    int rawSize = 0;
    uint8_t* rawImage = captureImage(&rawSize);
    if (!rawImage || rawSize == 0) {
        Serial.println("❌ Capture failed");
        return;
    }
    Serial.println("✅ Capture success");

    // Step 2: Compress
    Serial.println("Step 2: Compressing JPEG...");
    int jpegSize = 0;
    uint8_t* jpegData = jpegCompress(rawImage, rawSize, &jpegSize);
    if (!jpegData || jpegSize == 0) {
        Serial.println("❌ Compression failed");
        return;
    }
    Serial.println("✅ Compression success");

    // Step 3: Upload
    Serial.println("Step 3: Uploading to Pi...");
    if (!sendToPi(jpegData, jpegSize)) {
        Serial.println("❌ Upload failed");
        return;
    }
    Serial.println("✅ Upload success");

    Serial.println("=== ALL TESTS PASSED ===\n");
}
```

**Troubleshooting**:

| Problem | Solution |
|---------|----------|
| "Connection failed" | Check Pi IP, network connectivity, WiFi signal |
| "Response timeout" | Pi may be busy, try again, check logs on Pi |
| "Invalid image file" | Flask error: image data corrupted, check multipart encoding |
| Server returns 400 | Missing device_id or image field, check form-data |
| Server returns 404 | /predict endpoint doesn't exist on Pi, check Flask app |

**✅ Success Criteria**:
- Connection established to Pi
- HTTP request sent with valid multipart body
- Server responds with 200 OK or valid JSON
- Response contains `"is_fake"` field
- Multiple uploads work without memory leaks
- LED feedback shows correct result (red=fake, green=real)

---

## Phase 2: Backend Robustness

**Timeline**: 4-5 hours
**Status**: NOT STARTED
**Files Modified**: `app.py`, `requirements.txt`

### Step 2.1: Add Request Timeouts

**Time**: 30 minutes
**Priority**: 🟡 HIGH
**File**: `deployment/pi_server/app.py`

**Issue**: Webhook and email requests can hang indefinitely if network is slow

**Implementation**:

```python
# In send_alerts function around line 298

def _send():
    """Send alert via webhook or email."""
    try:
        response = requests.post(
            CONFIG['WEBHOOK_URL'],
            json=payload,
            timeout=5  # ✅ Add 5 second timeout
        )
        response.raise_for_status()  # Raise exception for bad status
        logger.info(f"Webhook sent successfully: {response.status_code}")

    except requests.Timeout:
        logger.error("Webhook request timed out after 5 seconds")

    except requests.ConnectionError:
        logger.error("Failed to connect to webhook URL")

    except requests.RequestException as e:
        logger.error(f"Webhook request failed: {e}")
```

**✅ Success Criteria**:
- All `requests.post()` calls have timeout parameter
- Webhook requests don't hang > 5 seconds
- Timeout errors are logged properly

---

### Step 2.2: Add Rate Limiting

**Time**: 1 hour
**Priority**: 🟡 HIGH
**File**: `deployment/pi_server/app.py` + `requirements.txt`

**Issue**: No protection against DDoS or spam requests

**Step 1: Update Requirements**:

```bash
# Edit deployment/pi_server/requirements.txt
Flask==2.3.2
Werkzeug==2.3.6
onnxruntime==1.15.0
Pillow==10.0.0
requests==2.31.0
Flask-Limiter==3.5.0  # ✅ ADD THIS LINE
```

**Install**:
```bash
pip install Flask-Limiter==3.5.0
```

**Step 2: Add to app.py**:

```python
# After line 35, add:
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="memory://"
)

# Apply to /predict endpoint (around line 334)
@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")  # ✅ Add this decorator
def predict():
    """Receive image from Nicla device and return prediction."""
    # ... existing code ...
```

**Rate Limits Strategy**:
- `/predict`: 10 req/min (inference-heavy endpoint)
- `/api/dashboard-data`: 60 req/min (allow frequent polling)
- Default: 100 req/hour (all other routes)

**✅ Success Criteria**:
- Rate limiter installed and imported
- /predict endpoint limited to 10 req/min
- Exceeding limit returns 429 Too Many Requests
- Can verify with: `ab -n 15 -c 1 http://localhost:5000/predict`

---

### Step 2.3: Input Validation

**Time**: 1 hour
**Priority**: 🟡 HIGH
**File**: `deployment/pi_server/app.py`

**Issues**:
- device_id not sanitized (path traversal risk)
- Image size not validated (DoS risk)
- Image format not verified (corruption risk)

**Implementation**:

```python
# In predict() function, add after line 360:

import re
import os

# Step 1: Sanitize device_id
device_id = request.form.get('device_id', '')

if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', device_id):
    logger.warning(f"Invalid device_id attempted: {device_id}")
    return jsonify({'error': 'Invalid device_id format'}), 400

# Step 2: Validate image size before reading
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB

image_file = request.files.get('image')
if not image_file:
    return jsonify({'error': 'Missing image file'}), 400

# Check file size
image_file.seek(0, os.SEEK_END)
file_size = image_file.tell()
image_file.seek(0)

if file_size > MAX_IMAGE_SIZE:
    logger.warning(f"Image too large: {file_size} bytes from {device_id}")
    return jsonify({'error': f'Image too large (max 5 MB)'}), 400

# Step 3: Read and validate image format
image_data = image_file.read()

try:
    from PIL import Image
    import io

    # Verify it's a valid image
    img = Image.open(io.BytesIO(image_data))
    img.verify()  # Check integrity

    # Only allow common formats
    if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
        logger.warning(f"Unsupported format {img.format} from {device_id}")
        return jsonify({'error': f'Unsupported image format: {img.format}'}), 400

except Exception as e:
    logger.warning(f"Invalid image from {device_id}: {e}")
    return jsonify({'error': 'Invalid image file'}), 400

# Proceed with preprocessing...
```

**✅ Success Criteria**:
- device_id validation blocks invalid characters
- Image size check prevents oversized uploads
- Image format validation prevents corrupted files
- All three checks log warnings
- Valid images pass through unaffected

---

### Step 2.4: Thread Safety

**Time**: 1 hour
**Priority**: 🟡 HIGH
**File**: `deployment/pi_server/app.py`

**Issue**: Multiple Nicla devices sending simultaneously causes race conditions in device_stats

**Implementation**:

```python
# After imports, add around line 36:
from threading import Lock

stats_lock = Lock()

# Update log_prediction function (around line 217):
def log_prediction(device_id, prediction, image_path=None):
    """Log prediction and update device statistics (thread-safe)."""

    with stats_lock:  # Acquire lock
        # Initialize device if first time
        if device_id not in device_stats:
            device_stats[device_id] = {
                'total_images': 0,
                'fake_detections': 0,
                'last_prediction': None,
                'avg_fake_prob': 0.0
            }

        # Update statistics atomically
        device_stats[device_id]['total_images'] += 1

        if prediction['is_fake']:
            device_stats[device_id]['fake_detections'] += 1

        # Calculate running average
        total = device_stats[device_id]['total_images']
        old_avg = device_stats[device_id]['avg_fake_prob']
        new_prob = prediction['fake_probability']
        device_stats[device_id]['avg_fake_prob'] = \
            (old_avg * (total - 1) + new_prob) / total

        device_stats[device_id]['last_prediction'] = datetime.now().isoformat()

        # Save stats atomically
        save_stats()

    # Logging happens outside lock (don't hold lock during I/O)
    logger.info(f"[{device_id}] Prediction logged: fake={prediction['is_fake']}")

# Update save_stats function to also use lock:
def save_stats():
    """Save device statistics to JSON file (thread-safe)."""
    with stats_lock:
        try:
            with open(CONFIG['STATS_FILE'], 'w') as f:
                json.dump(device_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
```

**✅ Success Criteria**:
- Lock protects all device_stats access
- File writes are atomic
- No race conditions with concurrent requests
- Stats remain consistent

---

### Step 2.5: Disk Space Management

**Time**: 1 hour
**Priority**: 🟡 HIGH
**File**: `deployment/pi_server/app.py`

**Issue**: Suspicious images grow unbounded, can fill SD card

**Implementation**:

```python
# Add after line 199:
from pathlib import Path
import time

def cleanup_old_images(max_storage_mb=1000, max_age_days=30):
    """
    Clean up old suspicious images if storage exceeds limit.

    Args:
        max_storage_mb: Maximum storage (default 1 GB)
        max_age_days: Maximum image age (default 30 days)
    """
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

    logger.info(f"Storage: {total_size_mb:.1f} MB in {len(image_files)} images")

    # Delete old files if over limit
    if total_size_mb > max_storage_mb:
        logger.info(f"Storage limit exceeded ({total_size_mb:.1f} > {max_storage_mb})")

        # Sort by modification time (oldest first)
        image_files.sort(key=lambda f: f.stat().st_mtime)

        deleted = 0
        freed_mb = 0

        for img_file in image_files:
            # Stop when we reach 80% threshold
            if total_size_mb - freed_mb <= max_storage_mb * 0.8:
                break

            file_size = img_file.stat().st_size / (1024 * 1024)
            img_file.unlink()
            freed_mb += file_size
            deleted += 1

        logger.info(f"Cleanup: Deleted {deleted} files, freed {freed_mb:.1f} MB")

    # Also delete very old files
    current_time = time.time()
    age_threshold = max_age_days * 24 * 60 * 60

    deleted_old = 0
    for img_file in image_files:
        if current_time - img_file.stat().st_mtime > age_threshold:
            img_file.unlink()
            deleted_old += 1

    if deleted_old > 0:
        logger.info(f"Cleanup: Deleted {deleted_old} images older than {max_age_days} days")

# In predict() function, add before line 388:
# Cleanup old images every 100 predictions
if device_stats.get('_cleanup_counter', 0) % 100 == 0:
    cleanup_old_images(max_storage_mb=1000, max_age_days=30)

device_stats['_cleanup_counter'] = device_stats.get('_cleanup_counter', 0) + 1
```

**✅ Success Criteria**:
- Images deleted when storage > 1 GB
- Old images (>30 days) automatically removed
- Cleanup runs every 100 predictions
- Log shows cleanup operations

---

## Phase 3: Remote Capture Feature

**Timeline**: 7-8 hours
**Status**: NOT STARTED
**Files Modified**: `app.py`, `dashboard.html`, `deepfake_detector.ino`

This phase adds the ability for users to trigger Nicla devices to capture images on-demand from the dashboard.

### Step 3.1: Backend Command Queue

**Time**: 2-3 hours
**Priority**: 🟢 NEW FEATURE
**File**: `deployment/pi_server/app.py`

**Architecture**:
```
User clicks "📷 Capture" button
    ↓
Dashboard: POST /api/capture-request → {"device_id": "nicla_1"}
    ↓
Flask: Stores in capture_commands queue
    ↓
Nicla: Polls GET /api/get-command/nicla_1
    ↓
Flask: Returns command if queued
    ↓
Nicla: Executes capture immediately
    ↓
Dashboard: Auto-refresh shows new image
```

**Implementation**:

```python
# Add after line 38:
from collections import deque
from datetime import datetime, timedelta

# Command queue for remote capture requests
capture_commands = {}  # {device_id: {'timestamp': ..., 'requester': ...}}
command_lock = Lock()  # Thread-safe access

# Add new endpoints before line 443:

@app.route('/api/capture-request', methods=['POST'])
@limiter.limit("5 per minute")
def request_capture():
    """
    Request a specific Nicla device to capture an image.

    Request body: {"device_id": "nicla_1"}
    Response: {"status": "queued", "device_id": "nicla_1", "timestamp": "..."}
    """
    try:
        data = request.get_json() or {}
        device_id = data.get('device_id', '')

        # Validate device_id
        if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', device_id):
            return jsonify({'error': 'Invalid device_id'}), 400

        # Check device exists
        if device_id not in device_stats:
            return jsonify({'error': f'Device {device_id} not found'}), 404

        # Queue the command
        with command_lock:
            capture_commands[device_id] = {
                'timestamp': datetime.now().isoformat(),
                'requester': request.remote_addr
            }

        logger.info(f"Capture request queued for {device_id}")

        return jsonify({
            'status': 'queued',
            'device_id': device_id,
            'timestamp': capture_commands[device_id]['timestamp']
        }), 200

    except Exception as e:
        logger.error(f"Capture request error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-command/<device_id>', methods=['GET'])
@limiter.limit("20 per minute")
def get_command(device_id):
    """
    Check if there's a pending command for this device.

    Returns: {"command": "capture"} or {"command": null}
    """
    try:
        # Validate device_id
        if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', device_id):
            return jsonify({'error': 'Invalid device_id'}), 400

        with command_lock:
            if device_id in capture_commands:
                cmd = capture_commands.pop(device_id)  # Remove after reading
                logger.info(f"Command retrieved by {device_id}")

                return jsonify({
                    'command': 'capture',
                    'timestamp': cmd['timestamp']
                }), 200
            else:
                return jsonify({'command': None}), 200

    except Exception as e:
        logger.error(f"Get command error: {e}")
        return jsonify({'error': str(e)}), 500


def cleanup_expired_commands():
    """Remove commands older than 60 seconds."""
    with command_lock:
        now = datetime.now()
        expired = []

        for device_id, cmd in capture_commands.items():
            cmd_time = datetime.fromisoformat(cmd['timestamp'])
            if (now - cmd_time).total_seconds() > 60:
                expired.append(device_id)

        for device_id in expired:
            del capture_commands[device_id]
            logger.info(f"Expired command for {device_id}")


# Add background cleanup worker (after imports, around line 60):
def command_cleanup_worker():
    """Background worker to cleanup expired commands."""
    while True:
        time.sleep(30)
        cleanup_expired_commands()

# Start on app startup (after limiter init):
cleanup_thread = Thread(target=command_cleanup_worker, daemon=True)
cleanup_thread.start()
```

**Testing Endpoints**:

```bash
# Request a capture from nicla_1
curl -X POST http://localhost:5000/api/capture-request \
  -H "Content-Type: application/json" \
  -d '{"device_id": "nicla_1"}'

# Response:
# {"status": "queued", "device_id": "nicla_1", "timestamp": "2026-01-04T..."}

# Check for command (Nicla would do this)
curl http://localhost:5000/api/get-command/nicla_1

# Response (when command queued):
# {"command": "capture", "timestamp": "2026-01-04T..."}

# Response (when no command):
# {"command": null}
```

**✅ Success Criteria**:
- /api/capture-request endpoint accepts POST requests
- Commands are stored thread-safely
- /api/get-command retrieves commands
- Commands expire after 60 seconds
- Rate limiting works (5 req/min for capture-request)

---

### Step 3.2: Nicla Polling Implementation

**Time**: 2-3 hours
**Priority**: 🟢 NEW FEATURE
**File**: `deployment/nicla/deepfake_detector.ino`

**Implementation**:

```cpp
// Add after line 57:
const int COMMAND_CHECK_INTERVAL = 3000;  // Check every 3 seconds
unsigned long lastCommandCheck = 0;

// Add new function after line 414:
bool checkForCommands() {
    /*
    Poll Flask server for pending commands.
    Called every 3 seconds from loop().
    Returns true if capture command received.
    */

    if (!client.connect(PI_SERVER_IP, PI_SERVER_PORT)) {
        return false;  // Silent fail, will retry next poll
    }

    // Send GET request
    String url = "/api/get-command/" + DEVICE_ID;
    client.print("GET ");
    client.print(url);
    client.println(" HTTP/1.1");
    client.print("Host: ");
    client.println(PI_SERVER_IP);
    client.println("Connection: close");
    client.println();

    // Wait for response (5 second timeout)
    unsigned long timeout = millis();
    while (client.connected() && !client.available()) {
        if (millis() - timeout > 5000) {
            client.stop();
            return false;
        }
        delay(10);
    }

    // Read response
    String response = "";
    while (client.available()) {
        response += (char)client.read();
    }
    client.stop();

    // Parse JSON: look for "command":"capture"
    if (response.indexOf("\"command\":\"capture\"") > 0) {
        Serial.println("✅ Remote capture command received!");
        return true;
    }

    return false;
}

// Update loop() function (around line 112):
void loop() {
    unsigned long currentMillis = millis();

    // Check for remote capture commands every 3 seconds
    if (currentMillis - lastCommandCheck >= COMMAND_CHECK_INTERVAL) {
        lastCommandCheck = currentMillis;

        if (checkForCommands()) {
            Serial.println("🎬 Executing remote capture...");
            captureAndSend();  // Immediate capture
        }
    }

    // Regular periodic capture (existing logic)
    if (currentMillis - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
        lastCaptureTime = currentMillis;
        captureAndSend();
    }

    delay(100);
}
```

**Testing**:

```cpp
// Add to serial monitor to test:
// Type 'C' to trigger a capture command
void serialTest() {
    if (Serial.available()) {
        char cmd = Serial.read();

        if (cmd == 'C' || cmd == 'c') {
            Serial.println("🎬 Manual capture test...");
            captureAndSend();
        }
    }
}

// Add to loop():
serialTest();
```

**✅ Success Criteria**:
- Nicla polls every 3 seconds
- Responds to remote capture commands
- Executes capture immediately when commanded
- No performance impact on periodic captures

---

### Step 3.3: Dashboard Capture UI

**Time**: 2 hours
**Priority**: 🟢 NEW FEATURE
**File**: `deployment/pi_server/dashboard.html`

**Implementation**: [SEE DEPLOYMENT_GUIDE_UI.md - Generated Separately]

Update `updateDeviceList()` function to add capture button to each device card, and add JavaScript functions for capture and toast notifications.

**✅ Success Criteria**:
- Capture button appears on each device card
- Button click sends POST request to /api/capture-request
- Toast notifications show feedback
- Button disabled during request
- Image appears in dashboard within 10 seconds

---

## Deployment Checklist

### Pre-Deployment

- [ ] Phase 1 Complete
  - [ ] ONNX model exported
  - [ ] Nicla image capture tested
  - [ ] Nicla JPEG compression tested
  - [ ] Nicla HTTP upload tested
  - [ ] End-to-end: Nicla → Pi works

- [ ] Phase 2 Complete
  - [ ] Request timeouts added
  - [ ] Rate limiting installed and tested
  - [ ] Input validation implemented
  - [ ] Thread safety verified
  - [ ] Disk cleanup tested

- [ ] Phase 3 Complete
  - [ ] Backend command queue working
  - [ ] Nicla polling functional
  - [ ] Dashboard capture UI ready
  - [ ] End-to-end remote capture works

### Deployment to Raspberry Pi

1. **Copy files to Pi**:
```bash
rsync -avz --delete --exclude='__pycache__' --exclude='*.pyc' \
  /home/incharaj/Team-Converge/deepfake-patch-audit/deployment/ \
  pi@192.168.1.100:/home/pi/deployment/
```

2. **Install dependencies**:
```bash
ssh pi@192.168.1.100
cd ~/deployment/pi_server
pip install -r ../requirements.txt
```

3. **Run Flask server**:
```bash
python3 app.py
```

4. **Verify dashboard**:
   - Open browser: http://192.168.1.100:5000
   - Should show metrics and device list

### Nicla Deployment

1. Upload Arduino sketch to all 4 Nicla devices
2. Configure WiFi SSID/password
3. Configure Pi server IP address
4. Set unique device_id for each (nicla_1, nicla_2, nicla_3, nicla_4)
5. Verify each connects and sends images

---

## Testing Procedures

### Test 1: Basic Inference

```bash
# Upload test image
curl -X POST http://localhost:5000/predict \
  -F "device_id=test" \
  -F "image=@test_image.jpg"

# Expected response:
# {"device_id": "test", "is_fake": false, "fake_probability": 0.15, ...}
```

### Test 2: Rate Limiting

```bash
# Try 15 requests in rapid succession
for i in {1..15}; do
  curl -X POST http://localhost:5000/predict \
    -F "device_id=test" \
    -F "image=@test_image.jpg"
  echo "Request $i"
done

# Requests 11-15 should get 429 Too Many Requests
```

### Test 3: Remote Capture

```bash
# Terminal 1: Monitor logs
ssh pi@192.168.1.100
tail -f ~/deployment/deepfake_detections.log

# Terminal 2: Request capture
curl -X POST http://192.168.1.100:5000/api/capture-request \
  -H "Content-Type: application/json" \
  -d '{"device_id": "nicla_1"}'

# Expected: Nicla captures within 3-5 seconds
# Image appears in dashboard within 10 seconds total
```

### Test 4: Concurrent Uploads

```bash
# Start 4 uploads simultaneously (one from each Nicla)
for i in {1..4}; do
  curl -X POST http://localhost:5000/predict \
    -F "device_id=nicla_$i" \
    -F "image=@test_image$i.jpg" &
done
wait

# All should succeed without race conditions
# Stats should show all 4 devices with correct counts
```

---

## Success Criteria Summary

### Phase 1: Complete Architecture
- ✅ ONNX model loaded and running inference
- ✅ All 4 Niclas connect and send images
- ✅ Images processed within 300ms end-to-end
- ✅ Predictions stored and logged
- ✅ Dashboard shows real-time statistics

### Phase 2: Production Robustness
- ✅ Request timeouts prevent hanging
- ✅ Rate limiting protects against abuse
- ✅ Input validation blocks attacks
- ✅ Thread safety verified under load
- ✅ Disk space managed automatically

### Phase 3: Remote Capture
- ✅ Users can trigger captures from dashboard
- ✅ Niclas respond to commands within 5 seconds
- ✅ Images appear in dashboard within 10 seconds
- ✅ Multiple devices work independently
- ✅ All commands clean up properly

---

## Support & Troubleshooting

### Dashboard Not Loading

```bash
# Check if Flask is running
ps aux | grep python | grep app.py

# Check port 5000
netstat -tulpn | grep 5000

# View Flask logs
tail -f /tmp/flask.log
```

### Nicla Not Connecting

```bash
# On Nicla: Check serial output
# Should show WiFi SSID and IP address

# On Pi: Check for device in stats
curl http://localhost:5000/stats | python -m json.tool

# Ping from Pi
ping <nicla-ip>
```

### Images Not Processing

```bash
# Check model file
ls -lh deployment/pi_server/models/deepfake_detector.onnx

# Test model loading
python3 -c "import onnxruntime; ..."

# Check logs
tail -f deepfake_detections.log
```

---

**Document Version**: 1.0
**Last Updated**: January 4, 2026
**Status**: Implementation Guide (Ready to Execute)
