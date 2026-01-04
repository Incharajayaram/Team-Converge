# Nicla Vision Implementation Guide

## Phase 1.2: Image Capture Implementation

This guide provides the complete implementation for capturing images from the Nicla Vision camera.

---

## Hardware Overview

### Nicla Vision Camera Specs
- **Sensor**: OV7670 (2MP capable)
- **Default Resolution**: 320x240 (reduced from 2MP for memory efficiency)
- **Format**: RGB565 or JPEG (hardware-configurable)
- **Frame Rate**: Up to 30 FPS
- **Field of View**: ~90 degrees (wide angle)

### Memory Considerations
- **Frame Buffer Size**: 320×240 RGB565 = 153,600 bytes (~150 KB)
- **Available RAM**: ~256-512 KB total
- **Typical Usage**: Can hold 2-3 frames in memory

---

## Implementation: initializeCamera()

### What It Does
Initializes the OV7670 camera hardware for image capture.

### Key Points
1. **Called once** during setup()
2. **Hardware-specific** - relies on Nicla camera drivers
3. **Sets resolution** to 320×240
4. **Configures format** to RGB565 (raw frames)
5. **Returns** true on success, false on failure

### Required Include
```cpp
#include <OV7670.h>
```

### Code to Add

Replace lines 210-218 with:

```cpp
bool initializeCamera() {
    /*
    Initialize OV7670 camera for 2MP capture.

    Hardware: Arduino Nicla Vision with integrated camera
    Resolution: 320x240 (2MP reduced mode)
    Format: RGB565 (raw frames, 2 bytes per pixel)
    Framerate: Up to 30 FPS

    This function:
    1. Initializes camera hardware
    2. Sets resolution to 320x240
    3. Configures pixel format
    4. Applies default settings (brightness, contrast)

    Returns:
        true: Camera initialized successfully
        false: Camera initialization failed (hardware issue)
    */

    Serial.println("\n========== CAMERA INITIALIZATION ==========");

    // Step 1: Initialize camera hardware
    // CAMERA_R320x240 specifies 320x240 resolution
    // PIXFORMAT_RGB565 specifies 16-bit RGB color format
    // Third parameter is framebuffer count (1 buffer for memory efficiency)

    Serial.print("Initializing OV7670 camera... ");

    if (!Camera.begin(CAMERA_R320x240, PIXFORMAT_RGB565, 1)) {
        Serial.println("FAILED!");
        Serial.println("❌ Camera hardware not responding");
        Serial.println("   - Check USB power (requires 500mA)");
        Serial.println("   - Check I2C connections");
        Serial.println("   - Try restarting the device");
        return false;
    }

    Serial.println("SUCCESS");

    // Step 2: Configure camera parameters
    // These settings optimize for general image capture
    // Adjust based on lighting conditions:
    // - Low light: increase brightness
    // - Bright: decrease brightness
    // - Low contrast: increase contrast

    Serial.println("Configuring camera settings...");

    // Brightness: -2 to 2 (0 = default, -2 = darker, 2 = brighter)
    // For indoors with moderate lighting: 0 (default)
    Camera.setBrightness(0);
    Serial.println("  ✓ Brightness: 0 (default)");

    // Contrast: -2 to 2 (0 = default, -2 = low, 2 = high)
    // For general use: 0 (default)
    Camera.setContrast(0);
    Serial.println("  ✓ Contrast: 0 (default)");

    // Saturation: -2 to 2 (0 = default, -2 = grayscale, 2 = vivid colors)
    // For deepfake detection: 0 (neutral colors)
    Camera.setSaturation(0);
    Serial.println("  ✓ Saturation: 0 (default)");

    // Step 3: Print camera information
    Serial.println("\nCamera Specifications:");
    Serial.println("  Resolution: 320×240 pixels");
    Serial.println("  Format: RGB565 (16-bit color)");
    Serial.println("  Frame size: 153,600 bytes");
    Serial.println("  Memory: ~150 KB per frame");

    Serial.println("\n✅ Camera initialized successfully!");
    Serial.println("========== CAMERA READY ==========\n");

    return true;
}
```

---

## Implementation: captureImage()

### What It Does
Captures a single frame from the camera and returns a pointer to the frame buffer.

### Key Points
1. **Called repeatedly** during operation (every 3 seconds)
2. **Grabs frame** from camera hardware
3. **Returns pointer** to frame buffer on success
4. **Returns nullptr** on failure (no frame available)
5. **Sets imageSize** to frame length in bytes

### Code to Add

Replace lines 220-243 with:

```cpp
uint8_t* captureImage(int& imageSize) {
    /*
    Capture a single frame from the camera.

    This function:
    1. Requests a frame from the camera
    2. Waits for frame availability
    3. Returns pointer to frame data
    4. Sets imageSize output parameter

    Args:
        imageSize: Reference output parameter for frame size in bytes

    Returns:
        Pointer to frame buffer on success
        nullptr on failure

    Frame Format:
        - Resolution: 320×240 pixels
        - Format: RGB565 (2 bytes per pixel)
        - Total size: 153,600 bytes
        - Duration: Valid only until next capture

    Important:
        - Must use frame immediately (before next capture)
        - Do not store pointer for later use
        - Next capture overwrites buffer
    */

    // Record start time for performance monitoring
    unsigned long captureStart = millis();

    // Step 1: Attempt to grab frame from camera
    // Camera.grab() returns camera_fb_t structure containing frame data
    // This is the main hardware capture operation

    Serial.println("\n📷 Capturing image from camera...");

    camera_fb_t *frame = Camera.grab();

    // Step 2: Check if frame was captured successfully
    if (!frame) {
        Serial.println("❌ Failed to capture frame!");
        Serial.println("   Possible causes:");
        Serial.println("   - Camera not responding");
        Serial.println("   - Buffer memory full");
        Serial.println("   - Hardware timeout");

        imageSize = 0;
        return nullptr;
    }

    // Step 3: Extract frame size
    // frame->len contains size in bytes
    imageSize = frame->len;

    // Step 4: Verify frame size is reasonable
    // Expected: ~153,600 bytes for 320×240 RGB565
    // Allow ±20% variance for different capture modes
    const int EXPECTED_SIZE = 320 * 240 * 2;  // 153,600 bytes
    const int MIN_SIZE = EXPECTED_SIZE * 0.8;
    const int MAX_SIZE = EXPECTED_SIZE * 1.2;

    if (imageSize < MIN_SIZE || imageSize > MAX_SIZE) {
        Serial.print("⚠️  Unusual frame size: ");
        Serial.print(imageSize);
        Serial.println(" bytes");
        Serial.print("   Expected: ~");
        Serial.print(EXPECTED_SIZE);
        Serial.println(" bytes");
    }

    // Step 5: Calculate and report capture time
    unsigned long captureTime = millis() - captureStart;

    Serial.print("✅ Frame captured: ");
    Serial.print(imageSize);
    Serial.print(" bytes in ");
    Serial.print(captureTime);
    Serial.println(" ms");

    // Step 6: Log frame properties
    Serial.print("   Resolution: ");
    Serial.print(frame->width);
    Serial.print("×");
    Serial.print(frame->height);
    Serial.print(" pixels");
    if (frame->format == PIXFORMAT_RGB565) {
        Serial.println(" (RGB565)");
    } else if (frame->format == PIXFORMAT_JPEG) {
        Serial.println(" (JPEG)");
    } else {
        Serial.println(" (unknown format)");
    }

    // Step 7: Return pointer to frame buffer
    // Important: This pointer is valid only until the next capture
    // The frame buffer is reused for the next capture

    return frame->buf;
}
```

---

## Testing the Implementation

### Test 1: Basic Camera Initialization

**Goal**: Verify camera hardware is working

**Code to add temporarily in loop():**

```cpp
// Add this function to test camera initialization
void testCameraInitialization() {
    static unsigned long lastTest = 0;
    static bool tested = false;

    if (!tested) {
        tested = true;
        Serial.println("\n=== CAMERA INITIALIZATION TEST ===");

        if (initializeCamera()) {
            Serial.println("✅ TEST PASSED: Camera initialized");
            ledOn(LED_G, 2000);  // Green light for 2 seconds
        } else {
            Serial.println("❌ TEST FAILED: Camera did not initialize");
            while (1) {
                ledBlink(LED_R, 3);  // Blink red 3 times
                delay(1000);
            }
        }
    }
}

// Call in setup(), after initializeCamera():
// testCameraInitialization();
```

### Test 2: Single Frame Capture

**Goal**: Verify camera captures frames correctly

**Code to add temporarily in loop():**

```cpp
// Add this function to test frame capture
void testFrameCapture() {
    static unsigned long lastTest = 0;

    if (millis() - lastTest > 5000) {  // Every 5 seconds
        lastTest = millis();

        Serial.println("\n=== FRAME CAPTURE TEST ===");

        int frameSize = 0;
        uint8_t* frame = captureImage(frameSize);

        if (frame && frameSize > 0) {
            Serial.println("✅ TEST PASSED: Frame captured");
            Serial.print("   Size: ");
            Serial.print(frameSize);
            Serial.println(" bytes");

            // Verify first few bytes (should be valid image data)
            Serial.print("   First 4 bytes (hex): ");
            for (int i = 0; i < 4; i++) {
                Serial.print(frame[i], HEX);
                Serial.print(" ");
            }
            Serial.println();

            ledOn(LED_G, 500);  // Brief green flash
        } else {
            Serial.println("❌ TEST FAILED: Failed to capture frame");
            ledBlink(LED_R, 1);
        }
    }
}

// Call in loop() after camera initialization successful
// testFrameCapture();
```

### Test 3: Continuous Capture Loop

**Goal**: Verify sustained capture operation

**Code to add temporarily in setup():**

```cpp
void testContinuousCapture() {
    Serial.println("\n=== CONTINUOUS CAPTURE TEST ===");
    Serial.println("Capturing 10 frames...");

    for (int i = 0; i < 10; i++) {
        Serial.print("Frame ");
        Serial.print(i + 1);
        Serial.print("/10: ");

        int frameSize = 0;
        uint8_t* frame = captureImage(frameSize);

        if (frame && frameSize > 0) {
            Serial.println("✅");
        } else {
            Serial.println("❌ FAILED");
            return;
        }

        delay(500);  // 500ms between captures
    }

    Serial.println("\n✅ CONTINUOUS CAPTURE TEST PASSED");
}

// Call once in setup() after camera init:
// testContinuousCapture();
```

---

## Serial Output Examples

### Successful Initialization

```
========== CAMERA INITIALIZATION ==========
Initializing OV7670 camera... SUCCESS
Configuring camera settings...
  ✓ Brightness: 0 (default)
  ✓ Contrast: 0 (default)
  ✓ Saturation: 0 (default)

Camera Specifications:
  Resolution: 320×240 pixels
  Format: RGB565 (16-bit color)
  Frame size: 153,600 bytes
  Memory: ~150 KB per frame

✅ Camera initialized successfully!
========== CAMERA READY ==========

```

### Successful Frame Capture

```
📷 Capturing image from camera...
✅ Frame captured: 153600 bytes in 45 ms
   Resolution: 320×240 pixels (RGB565)
```

### Common Error Messages

**Error: Camera initialization failed**
```
Initializing OV7670 camera... FAILED!
❌ Camera hardware not responding
   - Check USB power (requires 500mA)
   - Check I2C connections
   - Try restarting the device
```

**Error: Frame capture failed**
```
📷 Capturing image from camera...
❌ Failed to capture frame!
   Possible causes:
   - Camera not responding
   - Buffer memory full
   - Hardware timeout
```

---

## Troubleshooting

### Camera Won't Initialize

**Symptoms**: "Camera initialization failed" message

**Solutions**:
1. **Check USB Power**
   - Camera requires 500mA
   - Ensure using proper USB power adapter
   - Try different USB port

2. **Check I2C Connection**
   - Verify camera ribbon cable is fully inserted
   - Check for bent pins
   - Try reseating the cable

3. **Restart Device**
   - Press reset button
   - Unplug and replug USB
   - Upload sketch again

### Frames Are Corrupted

**Symptoms**: Frame data looks wrong, unusual size, garbled data

**Solutions**:
1. **Reduce Capture Frequency**
   - Try longer delay between captures
   - Camera may need more time to prepare frame

2. **Check Memory**
   - Verify sufficient RAM available
   - Reduce buffer size if needed
   - Watch for memory leaks

3. **Adjust Camera Settings**
   - Try different brightness/contrast
   - Ensure lighting is adequate

### Slow Frame Capture

**Symptoms**: Capture takes >100ms, system lags

**Solutions**:
1. **Normal Operation**
   - Capture taking 40-60ms is expected
   - This is hardware-dependent
   - Not a problem for 3-second intervals

2. **Monitor Performance**
   - Serial output shows capture time
   - If consistently >100ms, check CPU load
   - Reduce other processing if needed

---

## Hardware Configuration Options

### Alternative Resolutions

If memory is limited, you can reduce resolution:

```cpp
// 160x120 (VGA)
Camera.begin(CAMERA_R160x120, PIXFORMAT_RGB565, 1);

// 96x96 (smaller, uses less memory)
Camera.begin(CAMERA_R96x96, PIXFORMAT_RGB565, 1);
```

**Memory Usage**:
- 320×240: 153,600 bytes
- 160×120: 38,400 bytes
- 96×96: 18,432 bytes

### Alternative Pixel Formats

```cpp
// JPEG compressed (much smaller)
// Output: ~10-15 KB per frame
Camera.begin(CAMERA_R320x240, PIXFORMAT_JPEG, 1);

// Grayscale (uses less memory)
// Output: 76,800 bytes per frame
Camera.begin(CAMERA_R320x240, PIXFORMAT_GRAYSCALE, 1);
```

---

## Next Steps

After completing this implementation:

1. **Test thoroughly** with both test functions
2. **Verify output** in serial monitor
3. **Monitor memory** usage
4. **Move to Step 1.3**: JPEG Compression

---

## Key Functions Reference

### Camera Initialization
```cpp
bool Camera.begin(
    framesize_t frame_size,      // CAMERA_R320x240
    pixformat_t pixel_format,     // PIXFORMAT_RGB565
    uint8_t bufferCount           // Usually 1
);
```

### Configure Camera
```cpp
void Camera.setBrightness(int8_t brightness);  // -2 to 2
void Camera.setContrast(int8_t contrast);      // -2 to 2
void Camera.setSaturation(int8_t saturation);  // -2 to 2
```

### Capture Frame
```cpp
camera_fb_t* Camera.grab();  // Returns frame or nullptr
```

### Frame Structure
```cpp
struct camera_fb_t {
    uint8_t *buf;        // Pointer to image buffer
    size_t len;          // Length in bytes
    size_t width;        // Frame width
    size_t height;       // Frame height
    pixformat_t format;  // Pixel format
    // ... other fields
};
```

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| **Initialization Time** | 1-2 seconds |
| **Frame Capture Time** | 40-60 ms |
| **Memory per Frame** | ~150 KB |
| **Max Frames/Sec** | 15-30 FPS |
| **Practical Interval** | 3+ seconds (for WiFi + processing) |

---

## Success Criteria

✅ Camera initializes without errors
✅ Frames capture successfully every 3 seconds
✅ Frame size is 150-160 KB (RGB565 at 320×240)
✅ No memory leaks during sustained operation
✅ Serial output shows "✅" messages
✅ LED feedback works (green = success, red = error)
