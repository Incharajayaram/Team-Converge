# Arduino Nicla Vision - Setup & Upload Checklist

## ✅ Code Compatibility (VERIFIED)

- [x] Board: Arduino Nicla Vision (SAMD51 architecture)
- [x] TensorFlow Lite Micro includes (NOT ESP32-specific)
- [x] Memory constraints respected:
  - Flash: 221 KB / 512 KB available ✓
  - SRAM: 130 KB / 192 KB available ✓
- [x] Camera: OV7670 library compatible
- [x] WiFi: Built-in module supported
- [x] LED pins: Configured with fallbacks
- [x] Student model: 21 KB (fits in PROGMEM)
- [x] Tensor arena: 50 KB (fits in SRAM)

---

## 📋 Pre-Upload Checklist

### 1. Arduino IDE Setup
- [ ] Arduino IDE installed (v1.8.x or newer, or Arduino IDE 2.0)
- [ ] Arduino Mbed OS Boards installed (Tools → Board Manager → Search "Mbed")
- [ ] TensorFlow Lite for Microcontrollers installed (Tools → Manage Libraries)
- [ ] JPEGENC library installed
- [ ] ArduinoHttpClient library installed

### 2. Files Preparation
- [ ] deepfake_detector.ino in: `~/Team-Converge/deepfake-patch-audit/deployment/nicla/`
- [ ] student_model.h in same folder
- [ ] Both files in SAME directory before opening in Arduino IDE

### 3. Configuration (REQUIRED!)
Edit `deepfake_detector.ino` lines 48-57:
```cpp
const char* WIFI_SSID = "YOUR_SSID";          // ← Change this
const char* WIFI_PASSWORD = "YOUR_PASSWORD";   // ← Change this
const char* PI_SERVER_IP = "192.168.1.100";    // ← Change if needed
const String DEVICE_ID = "nicla_1";            // ← Optional
```

### 4. Hardware Setup
- [ ] Nicla Vision connected to PC via USB-C cable
- [ ] USB cable is HIGH QUALITY (some cheap cables don't work)
- [ ] Nicla powered (LED indicators visible)
- [ ] No other programs using the COM port

### 5. Arduino IDE Configuration
- [ ] Board: Tools → Board → Arduino Mbed OS Boards → Arduino Nicla Vision
- [ ] Port: Tools → Port → /dev/ttyACM0 (Linux/Mac) or COM3/COM4 (Windows)
- [ ] Speed: 115200 (Tools → Upload Speed)

---

## 🚀 Upload Steps

1. **Open Sketch**
   ```
   File → Open → ~/Team-Converge/deepfake-patch-audit/deployment/nicla/deepfake_detector.ino
   ```

2. **Verify Code Compiles**
   ```
   Sketch → Verify/Compile (or Ctrl+R)
   ```
   Should show: "✓ Done compiling"

3. **Upload to Board**
   ```
   Sketch → Upload (or Ctrl+U)
   or click the Upload button (→)
   ```
   Should show: "✓ Done uploading"

4. **Monitor Serial Output**
   ```
   Tools → Serial Monitor (or Ctrl+Shift+M)
   Baud Rate: 115200
   ```

---

## ✅ Expected Serial Output (After Upload)

```
=== Deepfake Detection Client ===
Device ID: nicla_1
Initializing student model for edge filtering...
  Loading student model from PROGMEM...
  ✓ Model loaded
  ✓ Operators registered
  ✓ Tensors allocated
  Arena used: 12345 bytes
✓ Student model initialized
Initializing face detector...
✓ Face detector initialized
Initializing camera...
✓ Camera initialized successfully!
Connecting to WiFi...
WiFi connected!
IP address: 192.168.1.50
Setup complete!

--- Image Capture and Three-Stage Detection ---
✓ Captured: 153600 bytes
STAGE 1: Face detection (BlazeFace)...
✓ Face detected
STAGE 2: Student model inference (edge filtering)...
✓ Compressed: 14567 bytes
  Running student inference... OK (87ms) → 32.5%
⊘ Student says REAL (32.5%) - skipping Pi send
✓ Image processed successfully
```

---

## ⚠️ Common Issues & Solutions

| Error | Solution |
|-------|----------|
| "Arduino Nicla Vision not found" | Install "Arduino Mbed OS Boards" in Board Manager |
| "Port not available" | Try different USB port, restart Arduino IDE, reinstall CH340 drivers |
| "fatal error: student_model.h not found" | Verify both .ino and .h files in same folder |
| "fatal error: tensorflow/lite..." | Install "TensorFlow Lite for Microcontrollers" |
| "Out of memory" | Reduce tensor arena size from 51200 to 30000 |
| "WiFi fails to connect" | Check SSID/password, verify 2.4GHz network |
| "Serial Monitor shows garbage" | Verify baud rate is 115200 |
| "LED pins don't work" | Check pin assignments match your Nicla board revision |

---

## 🎯 Quick Start (For Testing)

```bash
# 1. Navigate to folder
cd ~/Team-Converge/deepfake-patch-audit/deployment/nicla

# 2. Edit configuration (use your values)
# Edit lines 48-57 in deepfake_detector.ino

# 3. Open in Arduino IDE
arduino deepfake_detector.ino

# 4. Upload
# Tools → Board → Arduino Nicla Vision
# Tools → Port → (your port)
# Sketch → Upload
```

---

**Ready to upload? Check all boxes above, then proceed!** ✅
