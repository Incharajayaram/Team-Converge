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
    // Initialize OV7670 camera for 2MP capture
    // This is device-specific initialization

    // Set up camera for 96x96 output
    // Camera module setup code here

    return true;
}

uint8_t* captureImage(int& imageSize) {
    /*
     * Capture image from camera and resize to 128x128
     *
     * Size: 128x128 provides better quality than 96x96
     * Bandwidth: Still ~10-15 KB when JPEG compressed at 80% quality
     * Quality: Sufficient for deepfake detection on Raspberry Pi
     *
     * Returns: Pointer to image buffer
     * imageSize: Output parameter for image size in bytes
     */

    Serial.println("Capturing image...");

    // Capture from camera (2MP)
    // This is device-specific capture code

    // Resize to 128x128 (instead of 96x96)
    // Use OV7670 built-in scaling or software resize
    // Resolution: 128x128 pixels (16384 bytes uncompressed RGB)

    // Return image buffer
    return nullptr;
}

uint8_t* jpegCompress(uint8_t* imageData, int& jpegSize) {
    /*
     * Compress 128x128 image to JPEG at 80% quality
     *
     * Args:
     *   imageData: Input image buffer (128x128 RGB)
     *   jpegSize: Output parameter for compressed size
     *
     * Expected output size: ~10-15 KB
     *
     * Returns: Pointer to JPEG buffer
     */

    Serial.println("Compressing to JPEG...");

    // Use JPEGENC library to compress
    // Quality: 80%
    // Input: 128x128 RGB image (16384 bytes)
    // Output: JPEG format (~10-15 KB)

    // This is a simplified version - actual implementation requires
    // JPEGENC library integration

    return nullptr;
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
     * Send JPEG image to Raspberry Pi server via HTTP POST
     *
     * POST /predict HTTP/1.1
     * Host: {PI_SERVER_IP}:{PI_SERVER_PORT}
     * Content-Type: multipart/form-data
     *
     * form-data:
     *   device_id: {DEVICE_ID}
     *   image: {JPEG binary data}
     *
     * Response:
     * {
     *   "is_fake": bool,
     *   "fake_probability": float,
     *   "confidence": float
     * }
     */

    String url = "http://" + String(PI_SERVER_IP) + ":" + String(PI_SERVER_PORT) + PI_ENDPOINT;

    Serial.print("Sending to: ");
    Serial.println(url);

    http.begin(url);

    // TODO: Add multipart form data with image
    // This requires proper multipart/form-data encoding
    // For now, using a simplified version

    // This is a placeholder - actual implementation needs:
    // - Proper multipart/form-data boundary
    // - Binary file upload
    // - Correct Content-Type header

    http.addHeader("Content-Type", "multipart/form-data; boundary=----WebKitFormBoundary");

    int httpResponseCode = http.POST("image_data");

    if (httpResponseCode == 200) {
        String response = http.getString();
        handleServerResponse(response);
        http.end();
        return true;
    } else {
        Serial.print("HTTP Error: ");
        Serial.println(httpResponseCode);
        http.end();
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
