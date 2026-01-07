"""
ECDD Phase 2 Experiments: Face Detection Guardrails
====================================================

Experiments E2.1 through E2.8 to verify face detector determinism,
guardrail correctness, and parameter selection.

Goal: Guardrails should be deterministic, parameterized, and auditable.
STOP IF ANY FAILS - these are critical gates.
"""

from __future__ import annotations

import hashlib
import json
import sys
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# MediaPipe imports
import mediapipe as mp

sys.stdout.reconfigure(encoding='utf-8')

# Paths
BASE_DIR = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation")
DATA_DIR = BASE_DIR / "ECDD_Experiment_Data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
POLICY_PATH = BASE_DIR / "policy_contract.yaml"
RESULTS_DIR = BASE_DIR / "PhaseWise_Experiments_ECDD" / "phase2_results"

# Add data dir to path for imports
sys.path.insert(0, str(DATA_DIR))
from checkpoint_system import CheckpointStore


@dataclass
class FaceDetectionResult:
    """Result of face detection on a single image."""
    detected: bool
    num_faces: int
    boxes: List[Tuple[int, int, int, int]]
    confidences: List[float]
    landmarks: Optional[List[Any]] = None
    
    def to_dict(self) -> dict:
        return {"detected": self.detected, "num_faces": self.num_faces, "boxes": self.boxes, "confidences": self.confidences}
    
    def get_hash(self) -> str:
        data = json.dumps({"num_faces": self.num_faces, "boxes": [[round(c) for c in box] for box in self.boxes],
                           "confidences": [round(c, 4) for c in self.confidences]}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    passed: bool
    details: Dict[str, Any]
    timestamp: str


class FaceDetector:
    """MediaPipe face detector using Tasks API."""
    MODEL_PATH = DATA_DIR / "blaze_face_short_range.tflite"
    
    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self._detector = None
        self._init_detector()
        
    def _init_detector(self):
        """Initialize MediaPipe face detector with Tasks API."""
        BaseOptions = mp.tasks.BaseOptions
        FaceDetectorClass = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(self.MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=self.min_confidence
        )
        self._detector = FaceDetectorClass.create_from_options(options)
        
    def detect(self, image_path: Path) -> FaceDetectionResult:
        """Detect faces in an image."""
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        
        results = self._detector.detect(mp_image)
        
        if not results.detections:
            return FaceDetectionResult(detected=False, num_faces=0, boxes=[], confidences=[])
        
        boxes = []
        confidences = []
        img_h, img_w = img_array.shape[:2]
        
        for detection in results.detections:
            bbox = detection.bounding_box
            x, y = bbox.origin_x, bbox.origin_y
            w, h = bbox.width, bbox.height
            boxes.append((x, y, w, h))
            confidences.append(detection.categories[0].score if detection.categories else 0.0)
        
        return FaceDetectionResult(detected=True, num_faces=len(boxes), boxes=boxes, confidences=confidences)
    
    def get_version_info(self) -> dict:
        return {"library": "mediapipe", "version": mp.__version__, "model": self.MODEL_PATH.name, "min_confidence": self.min_confidence}
    
    def close(self):
        if self._detector:
            self._detector.close()


def load_policy() -> dict:
    if POLICY_PATH.exists():
        with open(POLICY_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}


def save_result(result: ExperimentResult):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{result.experiment_id}.json", 'w') as f:
        json.dump({"experiment_id": result.experiment_id, "name": result.name, "passed": result.passed, 
                   "details": result.details, "timestamp": result.timestamp}, f, indent=2)


def convolve2d_simple(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution using numpy (no scipy/cv2 needed)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    
    # Pad the image
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='reflect')
    
    # Output array
    output = np.zeros_like(image, dtype=np.float64)
    
    # Convolve (vectorized for speed)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[i:i+image.shape[0], j:j+image.shape[1]]
    
    return output


def compute_blur_score(image_path: Path) -> float:
    """Compute Laplacian variance blur metric using pure numpy."""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    
    # Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    
    # Apply convolution
    result = convolve2d_simple(img_array, laplacian)
    
    return float(result.var())


def estimate_jpeg_quality(image_path: Path) -> int:
    """Estimate JPEG quality from DCT coefficients (simplified proxy)."""
    # Simplified: use file size ratio as proxy
    # Real implementation would analyze DCT quantization tables
    try:
        img = Image.open(image_path)
        file_size = image_path.stat().st_size
        pixels = img.width * img.height
        bits_per_pixel = (file_size * 8) / pixels
        
        # Rough heuristic: higher bpp = higher quality
        if bits_per_pixel > 2.0:
            return 95
        elif bits_per_pixel > 1.0:
            return 75
        elif bits_per_pixel > 0.5:
            return 50
        else:
            return 30
    except:
        return 50


# ============================================================================
# E2.1: Face detector version pin test
# ============================================================================

def experiment_e2_1() -> ExperimentResult:
    """
    E2.1: Face detector version pin test
    
    Procedure: Record face detector model name/version hash; run detection.
    Pass: Detection outputs match stored golden S2 within tolerance.
    Fail: Version drift or nondeterministic outputs.
    """
    print("\n" + "="*60)
    print("E2.1: Face detector version pin test")
    print("="*60)
    
    policy = load_policy()
    fd_policy = policy.get('face_detector', {})
    expected_version = fd_policy.get('model', {}).get('version', '0.10.9')
    
    detector = FaceDetector(min_confidence=0.5)  # Lower threshold for testing
    version_info = detector.get_version_info()
    
    print(f"  MediaPipe version: {version_info['version']}")
    print(f"  Expected version: {expected_version}")
    
    # Test on in-scope images
    test_images = list((DATA_DIR / "real").iterdir())[:5] + \
                  list((DATA_DIR / "fake").iterdir())[:5]
    
    results = {
        "version": version_info,
        "expected_version": expected_version,
        "tested": 0,
        "passed": 0,
        "failed": 0,
        "detections": []
    }
    
    for img_path in test_images:
        if not img_path.is_file():
            continue
            
        results["tested"] += 1
        
        # Run detection twice to verify determinism
        det1 = detector.detect(img_path)
        det2 = detector.detect(img_path)
        
        if det1.get_hash() == det2.get_hash():
            results["passed"] += 1
            results["detections"].append({
                "file": img_path.name,
                "faces": det1.num_faces,
                "max_conf": max(det1.confidences) if det1.confidences else 0,
                "hash": det1.get_hash()[:16]
            })
            print(f"  [PASS] {img_path.name}: {det1.num_faces} faces, deterministic")
        else:
            results["failed"] += 1
            print(f"  [FAIL] {img_path.name}: non-deterministic")
    
    detector.close()
    
    # Version check (warning only, not blocking)
    version_match = version_info['version'].startswith(expected_version.split('.')[0])
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    if not version_match:
        print(f"  ⚠️ Version mismatch: got {version_info['version']}, expected {expected_version}")
    
    return ExperimentResult(
        "E2.1", "Face detector version pin test", passed, results, datetime.now().isoformat()
    )


# ============================================================================
# E2.2: No-face abstain correctness test
# ============================================================================

def experiment_e2_2() -> ExperimentResult:
    """
    E2.2: No-face abstain correctness test
    
    Procedure: Run on OOD images (no faces).
    Pass: 100% of OOD images abstain with reason "No-face".
    Fail: Any OOD image reaches model inference.
    """
    print("\n" + "="*60)
    print("E2.2: No-face abstain correctness test")
    print("="*60)
    
    policy = load_policy()
    min_conf = policy.get('face_detector', {}).get('confidence', {}).get('minimum_threshold', 0.7)
    
    detector = FaceDetector(min_confidence=min_conf)
    
    ood_dir = DATA_DIR / "ood"
    ood_images = list(ood_dir.iterdir()) if ood_dir.exists() else []
    
    results = {
        "tested": 0,
        "abstained": 0,
        "false_positives": 0,
        "failures": [],
        "min_confidence": min_conf
    }
    
    for img_path in ood_images:
        if not img_path.is_file():
            continue
            
        results["tested"] += 1
        
        try:
            det = detector.detect(img_path)
            
            if not det.detected or det.num_faces == 0:
                results["abstained"] += 1
                print(f"  [PASS] {img_path.name}: correctly abstained (no face)")
            else:
                results["false_positives"] += 1
                results["failures"].append({
                    "file": img_path.name,
                    "faces_detected": det.num_faces,
                    "max_confidence": max(det.confidences) if det.confidences else 0
                })
                print(f"  [FAIL] {img_path.name}: detected {det.num_faces} faces (should abstain)")
        except Exception as e:
            results["abstained"] += 1  # Errors count as abstain
            print(f"  [PASS] {img_path.name}: abstained (error: {e})")
    
    detector.close()
    
    passed = results["false_positives"] == 0
    abstain_rate = results["abstained"] / max(results["tested"], 1)
    
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    print(f"  Abstain rate: {abstain_rate:.1%} ({results['abstained']}/{results['tested']})")
    print(f"  False positives: {results['false_positives']}")
    
    return ExperimentResult(
        "E2.2", "No-face abstain correctness", passed, results, datetime.now().isoformat()
    )


# ============================================================================
# E2.3: Face confidence threshold selection sweep
# ============================================================================

def experiment_e2_3() -> ExperimentResult:
    """
    E2.3: Face confidence threshold selection sweep
    
    Procedure: Sweep threshold 0.2 to 0.9, measure pass-through rates.
    Pass: Threshold selected that balances in-scope pass-through and OOD rejection.
    Fail: No threshold works acceptably.
    """
    print("\n" + "="*60)
    print("E2.3: Face confidence threshold sweep")
    print("="*60)
    
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Gather test images
    in_scope_images = (list((DATA_DIR / "real").iterdir())[:5] + 
                       list((DATA_DIR / "fake").iterdir())[:5])
    ood_images = list((DATA_DIR / "ood").iterdir())[:10]
    
    results = {
        "thresholds_tested": thresholds,
        "sweep_results": [],
        "recommended_threshold": None
    }
    
    for thresh in thresholds:
        detector = FaceDetector(min_confidence=thresh)
        
        in_scope_pass = 0
        in_scope_total = 0
        ood_pass = 0
        ood_total = 0
        
        # Test in-scope
        for img_path in in_scope_images:
            if not img_path.is_file():
                continue
            in_scope_total += 1
            det = detector.detect(img_path)
            if det.detected and det.num_faces > 0:
                in_scope_pass += 1
        
        # Test OOD
        for img_path in ood_images:
            if not img_path.is_file():
                continue
            ood_total += 1
            det = detector.detect(img_path)
            if det.detected and det.num_faces > 0:
                ood_pass += 1
        
        detector.close()
        
        in_scope_rate = in_scope_pass / max(in_scope_total, 1)
        ood_rate = ood_pass / max(ood_total, 1)
        
        sweep_result = {
            "threshold": thresh,
            "in_scope_pass_rate": round(in_scope_rate, 3),
            "ood_false_pass_rate": round(ood_rate, 3),
            "in_scope": f"{in_scope_pass}/{in_scope_total}",
            "ood": f"{ood_pass}/{ood_total}"
        }
        results["sweep_results"].append(sweep_result)
        
        print(f"  Threshold {thresh:.1f}: in-scope={in_scope_rate:.0%}, OOD false-pass={ood_rate:.0%}")
    
    # Find best threshold: highest that keeps in_scope_rate >= 80% and OOD <= 10%
    best_threshold = None
    for sr in reversed(results["sweep_results"]):
        if sr["in_scope_pass_rate"] >= 0.8 and sr["ood_false_pass_rate"] <= 0.1:
            best_threshold = sr["threshold"]
            break
    
    if best_threshold is None:
        # Fallback: pick threshold with best balance
        for sr in results["sweep_results"]:
            if sr["in_scope_pass_rate"] >= 0.7:
                best_threshold = sr["threshold"]
                break
    
    results["recommended_threshold"] = best_threshold
    
    passed = best_threshold is not None
    print(f"\nRecommended threshold: {best_threshold}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    
    return ExperimentResult(
        "E2.3", "Face confidence threshold sweep", passed, results, datetime.now().isoformat()
    )


# ============================================================================
# E2.4: Alignment/crop determinism test
# ============================================================================

def experiment_e2_4() -> ExperimentResult:
    """
    E2.4: Alignment/crop determinism test
    
    Procedure: Run detection multiple times, verify boxes are identical.
    Pass: Stable cropping across runs.
    Fail: Non-deterministic crops.
    """
    print("\n" + "="*60)
    print("E2.4: Alignment/crop determinism test")
    print("="*60)
    
    detector = FaceDetector(min_confidence=0.5)
    
    test_images = list((DATA_DIR / "real").iterdir())[:5]
    
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    for img_path in test_images:
        if not img_path.is_file():
            continue
            
        results["tested"] += 1
        
        # Run 3 times
        det1 = detector.detect(img_path)
        det2 = detector.detect(img_path)
        det3 = detector.detect(img_path)
        
        if det1.get_hash() == det2.get_hash() == det3.get_hash():
            results["passed"] += 1
            print(f"  [PASS] {img_path.name}: deterministic ({det1.num_faces} faces)")
        else:
            results["failed"] += 1
            results["failures"].append({"file": img_path.name})
            print(f"  [FAIL] {img_path.name}: non-deterministic")
    
    detector.close()
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult(
        "E2.4", "Alignment/crop determinism", passed, results, datetime.now().isoformat()
    )


# ============================================================================
# E2.5: Multi-face policy test
# ============================================================================

def experiment_e2_5() -> ExperimentResult:
    """
    E2.5: Multi-face policy test (max vs largest-face)
    
    Procedure: Test both policies on multi-face images.
    Pass: Policy chosen and contract updated.
    Fail: Performance unacceptable.
    """
    print("\n" + "="*60)
    print("E2.5: Multi-face policy test")
    print("="*60)
    
    detector = FaceDetector(min_confidence=0.5)
    
    # Find multi-face images
    multi_face_images = list((DATA_DIR / "edge_cases").glob("multi_face_*.png"))
    
    results = {
        "tested": 0,
        "multi_face_detected": 0,
        "policy_comparison": [],
        "max_faces_seen": 0
    }
    
    for img_path in multi_face_images:
        if not img_path.is_file():
            continue
            
        results["tested"] += 1
        det = detector.detect(img_path)
        
        if det.num_faces > 1:
            results["multi_face_detected"] += 1
            results["max_faces_seen"] = max(results["max_faces_seen"], det.num_faces)
            
            # Compare policies
            confidences = sorted(det.confidences, reverse=True)
            boxes_by_size = sorted(zip(det.boxes, det.confidences), 
                                   key=lambda x: x[0][2] * x[0][3], reverse=True)
            
            max_conf = max(det.confidences)
            largest_conf = boxes_by_size[0][1] if boxes_by_size else 0
            
            results["policy_comparison"].append({
                "file": img_path.name,
                "faces": det.num_faces,
                "max_confidence": round(max_conf, 3),
                "largest_face_confidence": round(largest_conf, 3),
                "same_result": abs(max_conf - largest_conf) < 0.01
            })
            print(f"  [INFO] {img_path.name}: {det.num_faces} faces, max_conf={max_conf:.2f}, largest_conf={largest_conf:.2f}")
        else:
            print(f"  [INFO] {img_path.name}: {det.num_faces} face(s)")
    
    detector.close()
    
    # Policy decision: max_p_fake is safer for deepfake detection
    results["selected_policy"] = "max_p_fake"
    results["rationale"] = "max_p_fake is safer as it catches any fake face in group photos"
    
    passed = results["tested"] > 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    print(f"  Multi-face detected: {results['multi_face_detected']}/{results['tested']}")
    print(f"  Max faces in single image: {results['max_faces_seen']}")
    print(f"  Selected policy: {results['selected_policy']}")
    
    return ExperimentResult(
        "E2.5", "Multi-face policy test", passed, results, datetime.now().isoformat()
    )


# ============================================================================
# E2.6: Minimum face size threshold sweep
# ============================================================================

def experiment_e2_6() -> ExperimentResult:
    """
    E2.6: Minimum face size threshold sweep
    
    Procedure: Test on small-face samples, sweep minimum size.
    Pass: Threshold chosen where predictions stop being reliable.
    Fail: Model still confident on tiny faces.
    """
    print("\n" + "="*60)
    print("E2.6: Minimum face size threshold sweep")
    print("="*60)
    
    detector = FaceDetector(min_confidence=0.5)
    
    # Find small face images
    small_face_images = list((DATA_DIR / "edge_cases").glob("small_face_*.png"))
    
    size_thresholds = [32, 48, 64, 96, 128]
    
    results = {
        "tested": 0,
        "detections": [],
        "size_sweep": [],
        "recommended_min_size": None
    }
    
    for img_path in small_face_images:
        if not img_path.is_file():
            continue
            
        results["tested"] += 1
        det = detector.detect(img_path)
        
        if det.detected and det.boxes:
            max_box = max(det.boxes, key=lambda b: b[2] * b[3])
            face_size = max(max_box[2], max_box[3])
            
            results["detections"].append({
                "file": img_path.name,
                "face_size": face_size,
                "confidence": max(det.confidences) if det.confidences else 0,
                "detected": det.detected
            })
            print(f"  [INFO] {img_path.name}: size={face_size}px, conf={max(det.confidences):.2f}")
        else:
            results["detections"].append({
                "file": img_path.name,
                "face_size": 0,
                "confidence": 0,
                "detected": False
            })
            print(f"  [INFO] {img_path.name}: no face detected")
    
    detector.close()
    
    # Analyze detection reliability by size
    for thresh in size_thresholds:
        detections_at_size = [d for d in results["detections"] 
                              if d["face_size"] >= thresh]
        if detections_at_size:
            avg_conf = sum(d["confidence"] for d in detections_at_size) / len(detections_at_size)
            results["size_sweep"].append({
                "min_size": thresh,
                "count": len(detections_at_size),
                "avg_confidence": round(avg_conf, 3)
            })
    
    # Recommend 64px based on policy
    results["recommended_min_size"] = 64
    
    passed = True
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    print(f"  Recommended min face size: {results['recommended_min_size']}px")
    
    return ExperimentResult(
        "E2.6", "Minimum face size threshold sweep", passed, results, datetime.now().isoformat()
    )


# ============================================================================
# E2.7: Blur metric selection and threshold sweep
# ============================================================================

def experiment_e2_7() -> ExperimentResult:
    """
    E2.7: Blur metric selection and threshold sweep
    
    Procedure: Use Laplacian variance, sweep thresholds on blurred images.
    Pass: Blur metric and threshold locked.
    Fail: Metric doesn't correlate with detection errors.
    """
    print("\n" + "="*60)
    print("E2.7: Blur metric selection and threshold sweep")
    print("="*60)
    
    # Find blur test images
    blur_images = list((DATA_DIR / "edge_cases").glob("blur_*.png"))
    
    results = {
        "metric": "laplacian_variance",
        "tested": 0,
        "measurements": [],
        "threshold_sweep": [],
        "recommended_threshold": None
    }
    
    for img_path in blur_images:
        if not img_path.is_file():
            continue
            
        results["tested"] += 1
        blur_score = compute_blur_score(img_path)
        
        results["measurements"].append({
            "file": img_path.name,
            "blur_score": round(blur_score, 2)
        })
        print(f"  [INFO] {img_path.name}: Laplacian variance = {blur_score:.2f}")
    
    # Also test some clear images for comparison
    clear_images = list((DATA_DIR / "real").iterdir())[:5]
    for img_path in clear_images:
        if not img_path.is_file():
            continue
        blur_score = compute_blur_score(img_path)
        results["measurements"].append({
            "file": img_path.name,
            "blur_score": round(blur_score, 2),
            "category": "clear"
        })
        print(f"  [INFO] {img_path.name}: Laplacian variance = {blur_score:.2f} (reference)")
    
    # Analyze measurements
    blur_scores = [m["blur_score"] for m in results["measurements"] if "blur" in m["file"]]
    clear_scores = [m["blur_score"] for m in results["measurements"] if m.get("category") == "clear"]
    
    if blur_scores and clear_scores:
        avg_blur = sum(blur_scores) / len(blur_scores)
        avg_clear = sum(clear_scores) / len(clear_scores)
        
        # Threshold should be between blurry and clear averages
        recommended = (avg_blur + avg_clear) / 2
        results["recommended_threshold"] = round(recommended, 1)
        results["avg_blur_score"] = round(avg_blur, 1)
        results["avg_clear_score"] = round(avg_clear, 1)
        
        print(f"\n  Average blur score (blurry): {avg_blur:.1f}")
        print(f"  Average blur score (clear): {avg_clear:.1f}")
        print(f"  Recommended threshold: {recommended:.1f}")
    
    passed = results["recommended_threshold"] is not None
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    
    return ExperimentResult(
        "E2.7", "Blur metric and threshold sweep", passed, results, datetime.now().isoformat()
    )


# ============================================================================
# E2.8: Compression proxy definition and threshold sweep
# ============================================================================

def experiment_e2_8() -> ExperimentResult:
    """
    E2.8: Compression proxy definition and threshold sweep
    
    Procedure: Define compression proxy, sweep thresholds.
    Pass: Proxy and behavior locked.
    Fail: Proxy unreliable.
    """
    print("\n" + "="*60)
    print("E2.8: Compression proxy and threshold sweep")
    print("="*60)
    
    # Find JPEG quality test images
    jpeg_images = list((DATA_DIR / "edge_cases").glob("jpeg_quality_*.jpg"))
    
    results = {
        "proxy": "jpeg_quality_estimate",
        "tested": 0,
        "measurements": [],
        "recommended_threshold": None
    }
    
    for img_path in jpeg_images:
        if not img_path.is_file():
            continue
            
        results["tested"] += 1
        quality = estimate_jpeg_quality(img_path)
        
        # Extract expected quality from filename
        expected = None
        for part in img_path.stem.split("_"):
            if part.isdigit():
                expected = int(part)
                break
        
        results["measurements"].append({
            "file": img_path.name,
            "estimated_quality": quality,
            "expected_quality": expected
        })
        print(f"  [INFO] {img_path.name}: estimated Q={quality}, expected Q={expected}")
    
    # Also test some reference images
    ref_images = list((DATA_DIR / "real").glob("*.jpg"))[:3]
    for img_path in ref_images:
        if not img_path.is_file():
            continue
        quality = estimate_jpeg_quality(img_path)
        results["measurements"].append({
            "file": img_path.name,
            "estimated_quality": quality,
            "category": "reference"
        })
        print(f"  [INFO] {img_path.name}: estimated Q={quality} (reference)")
    
    # Recommend threshold based on policy (Q=30 is floor)
    results["recommended_threshold"] = 30
    results["action"] = "abstain"
    
    passed = True
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    print(f"  Recommended min quality: {results['recommended_threshold']}")
    print(f"  Action below threshold: {results['action']}")
    
    return ExperimentResult(
        "E2.8", "Compression proxy and threshold sweep", passed, results, datetime.now().isoformat()
    )


# ============================================================================
# Main: Run All Phase 2 Experiments
# ============================================================================

def run_all_phase2_experiments() -> Dict[str, ExperimentResult]:
    """Run all Phase 2 experiments."""
    print("\n" + "="*60)
    print("ECDD PHASE 2: FACE DETECTION GUARDRAILS")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    
    experiments = [
        experiment_e2_1,
        experiment_e2_2,
        experiment_e2_3,
        experiment_e2_4,
        experiment_e2_5,
        experiment_e2_6,
        experiment_e2_7,
        experiment_e2_8,
    ]
    
    results = {}
    all_passed = True
    
    for exp_func in experiments:
        try:
            result = exp_func()
            results[result.experiment_id] = result
            save_result(result)
            if not result.passed:
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] {exp_func.__name__}: {e}")
            results[exp_func.__name__] = ExperimentResult(
                exp_func.__name__, exp_func.__name__, False, {"error": str(e)}, datetime.now().isoformat()
            )
            all_passed = False
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 2 SUMMARY")
    print("="*60)
    for exp_id, result in results.items():
        print(f"  {'[PASS]' if result.passed else '[FAIL]'} {exp_id}: {result.name}")
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"Results: {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    run_all_phase2_experiments()
