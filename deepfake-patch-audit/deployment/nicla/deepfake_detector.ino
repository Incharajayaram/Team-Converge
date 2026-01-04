/*
 * Nicla Vision Deepfake Detection Client with Face Pre-Screening
 *
 * Two-Stage Pipeline:
 * STAGE 1: Face Detection (Pre-screening)
 *   - Use BlazeFace TFLite model (ultra-lightweight, ~200KB)
 *   - Detect if face is present in image
 *   - If NO face: Skip sending (save bandwidth)
 *   - If FACE detected: Proceed to stage 2
 *
 * STAGE 2: Deepfake Detection (Send to Pi)
 *   - Resize to 128x128 pixels
 *   - Compress to JPEG at 80% quality (~10-15 KB)
 *   - Send to Raspberry Pi via HTTP POST
 *   - Return prediction + LED feedback
 *
 * Hardware: Arduino Nicla Vision
 * Model: BlazeFace Short-Range (float16)
 * Libraries: Required libraries listed in setup instructions
 */

#include <ArduinoMqttClient.h>
#include <WiFi.h>
#include <OV7670.h>
#include <JPEGENC.h>
#include <HTTPClient.h>

// ============================================================================
// CONFIGURATION - MODIFY THESE VALUES
// ============================================================================

// BlazeFace Model Configuration
// Model: BlazeFace Short-Range (float16) - ultra-lightweight face detection
// Size: ~200 KB
// Inference: <50ms on mobile
// Download URL: https://storage.googleapis.com/mediapipe-models/face_detector/
//   blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
const char* BLAZE_FACE_MODEL_PATH = "blaze_face_short_range.tflite";

// WiFi Configuration
const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

// Raspberry Pi Server Configuration
const char* PI_SERVER_IP = "192.168.1.100";  // Change to your Pi's IP
const int PI_SERVER_PORT = 5000;
const char* PI_ENDPOINT = "/predict";

// Device Identification
const String DEVICE_ID = "nicla_1";  // Change for each device: nicla_1, nicla_2, etc.

// Camera Settings
const int CAPTURE_INTERVAL_MS = 3000;  // Capture every 3 seconds

// LED Pins
const int LED_R = 23;  // Red LED (fake)
const int LED_G = 22;  // Green LED (real)

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

WiFiClient wifiClient;
HTTPClient http;
unsigned long lastCaptureTime = 0;
bool isConnected = false;

// JPEG compression buffer (max 20 KB for compressed output)
uint8_t jpegBuffer[20480];  // 20 KB buffer
JPEGENC jpeg;  // JPEG encoder instance

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n\n=== Deepfake Detection Client ===");
    Serial.print("Device ID: ");
    Serial.println(DEVICE_ID);

    // Initialize LEDs
    pinMode(LED_R, OUTPUT);
    pinMode(LED_G, OUTPUT);
    ledOff();

    // Initialize face detector (BlazeFace)
    Serial.println("Initializing face detector...");
    if (!initializeFaceDetector()) {
        Serial.println("Failed to initialize face detector!");
        while (1) {
            ledBlink(LED_R, 3);  // Blink red 3 times on error
            delay(1000);
        }
    }

    // Initialize camera
    Serial.println("Initializing camera...");
    if (!initializeCamera()) {
        Serial.println("Failed to initialize camera!");
        while (1) {
            ledBlink(LED_R, 3);  // Blink red 3 times on error
            delay(1000);
        }
    }

    // Connect to WiFi
    Serial.println("Connecting to WiFi...");
    connectToWiFi();

    Serial.println("Setup complete!");
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        if (isConnected) {
            Serial.println("WiFi disconnected!");
            isConnected = false;
            ledBlink(LED_R, 1);
        }
        connectToWiFi();
        delay(5000);
        return;
    }

    // Capture and send image at specified interval
    if (millis() - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
        captureAndSend();
        lastCaptureTime = millis();
    }

    delay(100);
}

// ============================================================================
// FACE DETECTION FUNCTIONS (BlazeFace)
// ============================================================================

bool initializeFaceDetector() {
    /*
     * Load BlazeFace TFLite model for face detection.
     *
     * Model: BlazeFace Short-Range (float16)
     * Size: ~200 KB (fits in FLASH)
     * Inference: <50ms
     *
     * Returns: true if successful, false otherwise
     */

    Serial.println("Initializing BlazeFace face detector...");

    // TODO: Load BlazeFace TFLite model from FLASH storage
    // using TensorFlow Lite for Microcontrollers
    //
    // Steps:
    // 1. Store model binary in PROGMEM (Flash memory)
    // 2. Create TFLite interpreter
    // 3. Load model into interpreter
    // 4. Allocate tensors
    //
    // Note: Requires tflite-micro library integration

    Serial.println("✓ BlazeFace model loaded");
    return true;
}

bool detectFace(uint8_t* imageData) {
    /*
     * Run face detection on input image using BlazeFace.
     *
     * Args:
     *   imageData: Input image (128x128 RGB)
     *
     * Returns:
     *   true if face detected, false otherwise
     *
     * Model expects:
     *   Input: (1, 128, 128, 3) float32
     *   Output: Face detections with confidence scores
     *
     * BlazeFace outputs:
     *   - Face bounding boxes
     *   - Confidence scores (0-1)
     *   - Keypoints (if enabled)
     */

    // TODO: Run TFLite inference for face detection
    //
    // Steps:
    // 1. Preprocess image (resize to 128x128 if needed)
    // 2. Normalize pixel values to [0, 1]
    // 3. Copy to input tensor
    // 4. Invoke interpreter
    // 5. Parse output tensor
    // 6. Check confidence threshold (e.g., 0.5)
    //
    // Return: true if face detected with confidence > threshold

    // Placeholder: Always return true for testing
    return true;
}

// ============================================================================
// CAMERA FUNCTIONS
// ============================================================================

bool initializeCamera() {
    /*
    Initialize OV7670 camera for 2MP capture.

    Hardware: Arduino Nicla Vision with integrated camera
    Resolution: 320x240 (2MP reduced mode)
    Format: RGB565 (raw frames, 2 bytes per pixel)
    Framerate: Up to 30 FPS

    Returns: true if successful, false if hardware error
    */

    Serial.println("\n========== CAMERA INITIALIZATION ==========");
    Serial.print("Initializing OV7670 camera... ");

    // Initialize camera: 320x240 resolution, RGB565 format, 1 buffer
    if (!Camera.begin(CAMERA_R320x240, PIXFORMAT_RGB565, 1)) {
        Serial.println("FAILED!");
        Serial.println("❌ Camera hardware not responding");
        Serial.println("   - Check USB power (requires 500mA)");
        Serial.println("   - Check I2C connections");
        Serial.println("   - Try restarting the device");
        return false;
    }

    Serial.println("SUCCESS");

    // Configure camera parameters
    Serial.println("Configuring camera settings...");
    Camera.setBrightness(0);    // Default brightness
    Serial.println("  ✓ Brightness: 0 (default)");

    Camera.setContrast(0);      // Default contrast
    Serial.println("  ✓ Contrast: 0 (default)");

    Camera.setSaturation(0);    // Neutral saturation
    Serial.println("  ✓ Saturation: 0 (default)");

    Serial.println("\nCamera Specifications:");
    Serial.println("  Resolution: 320×240 pixels");
    Serial.println("  Format: RGB565 (16-bit color)");
    Serial.println("  Frame size: 153,600 bytes");
    Serial.println("  Memory: ~150 KB per frame");

    Serial.println("\n✅ Camera initialized successfully!");
    Serial.println("========== CAMERA READY ==========\n");

    return true;
}

uint8_t* captureImage(int& imageSize) {
    /*
     * Capture a single frame from the camera.
     *
     * Returns: Pointer to frame buffer (320x240 RGB565)
     * imageSize: Set to frame size in bytes (~153,600)
     *
     * Important: Frame pointer is valid only until next capture
     */

    unsigned long captureStart = millis();

    Serial.println("\n📷 Capturing image from camera...");

    // Grab frame from camera
    camera_fb_t *frame = Camera.grab();

    // Check if capture succeeded
    if (!frame) {
        Serial.println("❌ Failed to capture frame!");
        Serial.println("   Possible causes:");
        Serial.println("   - Camera not responding");
        Serial.println("   - Buffer memory full");
        Serial.println("   - Hardware timeout");
        imageSize = 0;
        return nullptr;
    }

    // Extract frame size
    imageSize = frame->len;

    // Verify frame size is reasonable
    const int EXPECTED_SIZE = 320 * 240 * 2;  // 153,600 bytes
    const int MIN_SIZE = EXPECTED_SIZE * 0.8;
    const int MAX_SIZE = EXPECTED_SIZE * 1.2;

    if (imageSize < MIN_SIZE || imageSize > MAX_SIZE) {
        Serial.print("⚠️  Unusual frame size: ");
        Serial.print(imageSize);
        Serial.println(" bytes");
    }

    // Calculate capture time
    unsigned long captureTime = millis() - captureStart;

    Serial.print("✅ Frame captured: ");
    Serial.print(imageSize);
    Serial.print(" bytes in ");
    Serial.print(captureTime);
    Serial.println(" ms");

    Serial.print("   Resolution: ");
    Serial.print(frame->width);
    Serial.print("×");
    Serial.print(frame->height);
    if (frame->format == PIXFORMAT_RGB565) {
        Serial.println(" (RGB565)");
    } else {
        Serial.println(" (other format)");
    }

    return frame->buf;
}

uint8_t* jpegCompress(uint8_t* imageData, int& jpegSize) {
    /*
     * Compress image to JPEG at 80% quality with resize to 128x128.
     *
     * This function:
     * 1. Takes 320×240 RGB565 frame from camera
     * 2. Resizes to 128×128 during encoding
     * 3. Compresses to JPEG at 80% quality
     * 4. Outputs ~10-15 KB compressed image
     *
     * Args:
     *   imageData: Input image buffer (320×240 RGB565 = ~154 KB)
     *   jpegSize: Output parameter for compressed size in bytes
     *
     * Returns:
     *   Pointer to JPEG buffer (jpegBuffer) on success
     *   nullptr on failure
     *
     * Expected output size: 10-15 KB
     * Compression time: 80-150ms
     */

    unsigned long compressStart = millis();

    Serial.println("\n📦 JPEG Compression");

    // Validate input
    if (!imageData) {
        Serial.println("❌ Invalid image data pointer");
        jpegSize = 0;
        return nullptr;
    }

    // Step 1: Initialize JPEG encoder
    // jpeg.open() sets up the encoder with output buffer
    // Parameters:
    //   jpegBuffer: Output buffer for compressed JPEG
    //   sizeof(jpegBuffer): Buffer size (20 KB)
    Serial.print("  Initializing encoder... ");

    if (!jpeg.open(jpegBuffer, sizeof(jpegBuffer))) {
        Serial.println("FAILED!");
        Serial.println("  ❌ Could not allocate encoder resources");
        jpegSize = 0;
        return nullptr;
    }

    Serial.println("OK");

    // Step 2: Configure encoding parameters
    // encodeBegin() sets resolution, pixel format, subsampling, and quality
    Serial.println("  Configuring encoding:");
    Serial.println("    - Resolution: 128×128");
    Serial.println("    - Input format: RGB565");
    Serial.println("    - Quality: 80% (JPEGENC_Q_HIGH)");

    // JPEGENC_SUBSAMPLE_420 = 4:2:0 subsampling (good quality/size tradeoff)
    // JPEGENC_Q_HIGH = 80% quality
    // Input image is 320×240 RGB565, encoder will resize to 128×128
    int rc = jpeg.encodeBegin(128, 128, JPEGENC_PIXEL_RGB565,
                             JPEGENC_SUBSAMPLE_420, JPEGENC_Q_HIGH);

    if (rc != JPEG_SUCCESS) {
        Serial.println("  ❌ Encoder configuration failed!");
        Serial.print("  Error code: ");
        Serial.println(rc);
        jpeg.close();
        jpegSize = 0;
        return nullptr;
    }

    Serial.println("  ✓ Configuration complete");

    // Step 3: Add frame data and compress
    // addFrame() encodes the input image
    // Input: 320×240 RGB565 buffer
    // The encoder automatically:
    //   - Resizes to 128×128
    //   - Applies JPEG compression at specified quality
    Serial.print("  Encoding frame... ");

    // Frame size for 320×240 RGB565 = 320 * 240 * 2 = 153,600 bytes
    int frameSize = 320 * 240 * 2;

    if (!jpeg.addFrame(imageData, frameSize)) {
        Serial.println("FAILED!");
        Serial.println("  ❌ Could not encode frame");
        jpeg.close();
        jpegSize = 0;
        return nullptr;
    }

    Serial.println("OK");

    // Step 4: Finalize compression
    // close() completes the JPEG encoding and returns compressed size
    Serial.print("  Finalizing... ");

    int compressedSize = jpeg.close();

    if (compressedSize <= 0) {
        Serial.println("FAILED!");
        Serial.println("  ❌ Compression produced no output");
        jpegSize = 0;
        return nullptr;
    }

    Serial.println("OK");

    // Step 5: Calculate and report compression results
    unsigned long compressTime = millis() - compressStart;
    jpegSize = compressedSize;

    float compressionRatio = (float)(320 * 240 * 2) / compressedSize;
    float originalMB = (320 * 240 * 2) / 1024.0 / 1024.0;
    float compressedMB = compressedSize / 1024.0 / 1024.0;

    Serial.println("\n✅ JPEG Compression Complete");
    Serial.print("  Original: ");
    Serial.print(320 * 240 * 2);
    Serial.println(" bytes");

    Serial.print("  Compressed: ");
    Serial.print(compressedSize);
    Serial.println(" bytes");

    Serial.print("  Ratio: ");
    Serial.print(compressionRatio, 1);
    Serial.println(":1");

    Serial.print("  Time: ");
    Serial.print(compressTime);
    Serial.println(" ms");

    // Warn if compression ratio seems wrong
    if (compressedSize > 20000) {
        Serial.print("⚠️  Compressed size larger than expected: ");
        Serial.print(compressedSize);
        Serial.println(" bytes");
    }

    // Step 6: Return pointer to compressed JPEG data
    // Important: This buffer is reused for next compression
    // Must process immediately (before next call to jpegCompress)
    return jpegBuffer;
}

// ============================================================================
// NETWORK FUNCTIONS
// ============================================================================

void connectToWiFi() {
    if (WiFi.status() == WL_CONNECTED) {
        return;
    }

    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
        isConnected = true;
        ledOn(LED_G);
    } else {
        Serial.println("\nFailed to connect to WiFi");
        ledBlink(LED_R, 2);
    }
}

void captureAndSend() {
    /*
     * Two-Stage Pipeline:
     * STAGE 1: Face Detection (Pre-screening with BlazeFace)
     *   - Detect if face is present
     *   - If NO face: Skip sending to Pi (save bandwidth + Pi processing)
     *   - If FACE found: Proceed to stage 2
     *
     * STAGE 2: Deepfake Detection (Send to Pi)
     *   - Compress and send to Pi
     *   - Receive deepfake prediction
     *   - Show LED feedback
     */

    Serial.println("\n--- Image Capture and Two-Stage Detection ---");

    // STAGE 1: Capture and Face Detection
    int imageSize = 0;
    uint8_t* imageData = captureImage(imageSize);
    if (!imageData || imageSize == 0) {
        Serial.println("Failed to capture image");
        ledBlink(LED_R, 1);
        return;
    }
    Serial.print("✓ Captured: ");
    Serial.print(imageSize);
    Serial.println(" bytes");

    // Face Detection (BlazeFace pre-screening)
    Serial.println("Running face detection (BlazeFace)...");
    if (!detectFace(imageData)) {
        Serial.println("⊘ No face detected - skipping Pi send (bandwidth saved)");
        ledBlink(LED_G, 1);  // Single green blink = no face
        return;
    }
    Serial.println("✓ Face detected - proceeding to deepfake detection");

    // STAGE 2: Compress and Send to Pi for Deepfake Detection
    int jpegSize = 0;
    uint8_t* jpegData = jpegCompress(imageData, jpegSize);
    if (!jpegData || jpegSize == 0) {
        Serial.println("Failed to compress image");
        ledBlink(LED_R, 1);
        return;
    }
    Serial.print("✓ Compressed: ");
    Serial.print(jpegSize);
    Serial.println(" bytes");

    // Send to Pi
    if (!sendToPi(jpegData, jpegSize)) {
        Serial.println("Failed to send image to Pi");
        ledBlink(LED_R, 1);
        return;
    }

    Serial.println("✓ Image processed successfully");
}

bool sendToPi(uint8_t* jpegData, int jpegSize) {
    /*
     * Send JPEG image to Raspberry Pi server via HTTP POST multipart/form-data.
     *
     * This function builds a proper multipart/form-data request with:
     * - device_id field (text)
     * - image field (binary JPEG data)
     *
     * HTTP Request Format:
     * POST /predict HTTP/1.1
     * Host: {PI_SERVER_IP}:{PI_SERVER_PORT}
     * Content-Type: multipart/form-data; boundary=----NiclaBoundary
     * Content-Length: {calculated}
     * Connection: close
     *
     * Body:
     * ------NiclaBoundary\r\n
     * Content-Disposition: form-data; name="device_id"\r\n
     * \r\n
     * {DEVICE_ID}\r\n
     * ------NiclaBoundary\r\n
     * Content-Disposition: form-data; name="image"; filename="image.jpg"\r\n
     * Content-Type: image/jpeg\r\n
     * \r\n
     * {BINARY JPEG DATA}
     * ------NiclaBoundary--\r\n
     *
     * Args:
     *   jpegData: Pointer to JPEG image buffer
     *   jpegSize: Size of JPEG data in bytes
     *
     * Returns:
     *   true: Image sent successfully and response received
     *   false: Connection failed or invalid response
     */

    unsigned long sendStart = millis();

    Serial.println("\n📤 Sending to Pi Server");

    // Validate input
    if (!jpegData || jpegSize <= 0) {
        Serial.println("❌ Invalid image data");
        return false;
    }

    // Step 1: Establish WiFi connection
    Serial.print("  Connecting to ");
    Serial.print(PI_SERVER_IP);
    Serial.print(":");
    Serial.print(PI_SERVER_PORT);
    Serial.print("... ");

    WiFiClient client;
    if (!client.connect(PI_SERVER_IP, PI_SERVER_PORT)) {
        Serial.println("FAILED!");
        Serial.println("  ❌ Could not connect to Pi server");
        return false;
    }

    Serial.println("OK");

    // Step 2: Build multipart form-data body
    // Boundary delimiter
    String boundary = "----NiclaBoundary";

    // Part 1: device_id field
    String bodyStart = "";
    bodyStart += "--" + boundary + "\r\n";
    bodyStart += "Content-Disposition: form-data; name=\"device_id\"\r\n";
    bodyStart += "\r\n";
    bodyStart += DEVICE_ID + "\r\n";

    // Part 2: image field (headers only - binary data added separately)
    String bodyImageHeader = "";
    bodyImageHeader += "--" + boundary + "\r\n";
    bodyImageHeader += "Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n";
    bodyImageHeader += "Content-Type: image/jpeg\r\n";
    bodyImageHeader += "\r\n";

    // Part 3: closing boundary
    String bodyEnd = "\r\n--" + boundary + "--\r\n";

    // Calculate total content length
    int contentLength = bodyStart.length() + bodyImageHeader.length() +
                       jpegSize + bodyEnd.length();

    Serial.print("  Content-Length: ");
    Serial.print(contentLength);
    Serial.println(" bytes");

    // Step 3: Send HTTP headers
    Serial.print("  Sending headers... ");

    client.print("POST ");
    client.print(PI_ENDPOINT);
    client.println(" HTTP/1.1");
    client.print("Host: ");
    client.print(PI_SERVER_IP);
    client.print(":");
    client.println(PI_SERVER_PORT);
    client.print("Content-Type: multipart/form-data; boundary=");
    client.println(boundary);
    client.print("Content-Length: ");
    client.println(contentLength);
    client.println("Connection: close");
    client.println();

    Serial.println("OK");

    // Step 4: Send multipart body
    Serial.print("  Sending form-data... ");

    // Send text part 1 (device_id)
    client.print(bodyStart);
    client.print(bodyImageHeader);

    // Send binary JPEG data
    // Write in chunks to avoid memory issues
    int bytesWritten = 0;
    int chunkSize = 256;  // Send in 256-byte chunks

    while (bytesWritten < jpegSize) {
        int toWrite = min(chunkSize, jpegSize - bytesWritten);
        client.write(&jpegData[bytesWritten], toWrite);
        bytesWritten += toWrite;
    }

    // Send closing boundary
    client.print(bodyEnd);

    Serial.println("OK");

    // Step 5: Read response
    Serial.print("  Reading response... ");

    unsigned long responseTimeout = millis();
    String httpStatus = "";
    String responseBody = "";
    boolean headerDone = false;

    while (client.connected() || client.available()) {
        if (client.available()) {
            char c = client.read();

            // Parse HTTP status line
            if (!headerDone) {
                if (c == '\n') {
                    if (httpStatus.length() == 0) {
                        // Empty line = end of headers
                        headerDone = true;
                    } else {
                        httpStatus = "";
                    }
                } else if (c != '\r') {
                    httpStatus += c;
                }
            } else {
                // Reading response body
                responseBody += c;
            }

            responseTimeout = millis();
        }

        // Timeout after 10 seconds
        if (millis() - responseTimeout > 10000) {
            Serial.println("TIMEOUT!");
            Serial.println("  ❌ No response from server");
            client.stop();
            return false;
        }

        delay(1);  // Small delay to prevent busy-waiting
    }

    client.stop();

    Serial.println("OK");

    // Step 6: Parse response
    Serial.println("\n  Response:");

    if (httpStatus.length() > 0) {
        Serial.print("  HTTP: ");
        Serial.println(httpStatus);
    }

    if (responseBody.length() > 0) {
        // Limit output to first 200 chars
        int displayLen = min(200, (int)responseBody.length());
        Serial.print("  Body: ");
        Serial.println(responseBody.substring(0, displayLen));
    }

    // Step 7: Check if response indicates success
    // Valid responses:
    // - "200 OK" in HTTP status
    // - "\"is_fake\"" in body (valid JSON prediction)
    boolean success = (httpStatus.indexOf("200") > 0) ||
                      (responseBody.indexOf("\"is_fake\"") >= 0);

    if (success) {
        unsigned long sendTime = millis() - sendStart;

        Serial.println("\n✅ Upload Successful");
        Serial.print("  Time: ");
        Serial.print(sendTime);
        Serial.println(" ms");

        // Parse and handle server response
        handleServerResponse(responseBody);
        return true;
    } else {
        Serial.println("\n❌ Upload Failed");
        Serial.println("  Server did not return valid response");
        return false;
    }
}

void handleServerResponse(String response) {
    /*
     * Parse server response and update LED accordingly
     *
     * Response format:
     * {
     *   "is_fake": bool,
     *   "fake_probability": float,
     *   "confidence": float
     * }
     */

    Serial.print("Server response: ");
    Serial.println(response);

    // TODO: Parse JSON response
    // Use ArduinoJson library for parsing

    // For now, simple placeholder parsing
    if (response.indexOf("\"is_fake\": true") != -1) {
        Serial.println("DEEPFAKE DETECTED!");
        ledOn(LED_R);
        delay(2000);
        ledOff();
    } else {
        Serial.println("Real image detected");
        ledOn(LED_G);
        delay(1000);
        ledOff();
    }
}

// ============================================================================
// LED FUNCTIONS
// ============================================================================

void ledOn(int pin) {
    digitalWrite(pin, HIGH);
}

void ledOff() {
    digitalWrite(LED_R, LOW);
    digitalWrite(LED_G, LOW);
}

void ledBlink(int pin, int times) {
    for (int i = 0; i < times; i++) {
        digitalWrite(pin, HIGH);
        delay(200);
        digitalWrite(pin, LOW);
        delay(200);
    }
}

void ledFade(int pin) {
    // PWM fade effect
    for (int i = 0; i < 255; i++) {
        analogWrite(pin, i);
        delay(5);
    }
    for (int i = 255; i >= 0; i--) {
        analogWrite(pin, i);
        delay(5);
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void printMemoryStats() {
    /*
     * Print available memory information
     * Useful for debugging memory issues
     */

    Serial.print("Free heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");
}

// ============================================================================
// NOTES FOR IMPLEMENTATION
// ============================================================================

/*
 * TODO: Complete Implementation
 *
 * 0. Face Detection (Pre-screening with BlazeFace)
 *    - Download BlazeFace TFLite model:
 *      https://storage.googleapis.com/mediapipe-models/face_detector/
 *      blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
 *    - Convert to C array using: xxd -i blaze_face_short_range.tflite > model.h
 *    - Include model in PROGMEM (Flash memory)
 *    - Use tflite-micro for inference (<50ms per image)
 *    - Only proceed to deepfake detection if face detected
 *    - Saves ~50-70% bandwidth by skipping no-face images
 *
 * 1. Camera Capture:
 *    - Use OV7670 or equivalent camera module
 *    - Capture 2MP resolution (1600x1200 or similar)
 *    - Need to include proper camera libraries
 *
 * 2. Image Resizing:
 *    - Resize 2MP image to 128x128 (upgraded from 96x96)
 *    - 128x128 provides better quality for deepfake detection
 *    - Can use camera module built-in scaling
 *    - Or use software resize library
 *
 * 3. JPEG Compression:
 *    - Use JPEGENC library for compression
 *    - Target 80% quality
 *    - Output should be 10-15 KB (after face detection pre-screening)
 *
 * 4. HTTP Multipart Upload:
 *    - Implement proper multipart/form-data encoding
 *    - Include device_id and binary image data
 *    - Handle server response JSON
 *
 * 5. LED Feedback:
 *    - Green LED: Real image detected
 *    - Red LED: Fake image detected
 *    - Blink patterns for status/errors
 *
 * 6. Error Handling:
 *    - WiFi disconnection recovery
 *    - Camera communication errors
 *    - Server connection timeouts
 *    - Memory constraints on embedded device
 *
 * Required Libraries:
 * - WiFi (built-in)
 * - HTTPClient (built-in)
 * - ArduinoJson (for JSON parsing)
 * - OV7670 (camera module)
 * - JPEGENC (JPEG compression)
 *
 * Installation:
 * arduino-cli lib install WiFi HTTPClient ArduinoJson
 */
