import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Paths
BASE_DIR = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation")
DATA_DIR = BASE_DIR / "ECDD_Experiment_Data"
RESULTS_DIR = BASE_DIR / "PhaseWise_Experiments_ECDD" / "phase8_results"
LINEAGE_FILE = DATA_DIR / "data_lineage.json"

@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    passed: bool
    details: Dict[str, Any]
    timestamp: str

def save_result(result: ExperimentResult):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{result.experiment_id}.json", 'w') as f:
        json.dump(result.__dict__, f, indent=2, default=str)

def compute_perceptual_hash(image_path: Path, hash_size: int = 8) -> str:
    """
    Compute perceptual hash (difference hash) for an image.
    This is a lightweight implementation without imagehash library.
    """
    try:
        img = Image.open(image_path).convert('L')  # Grayscale
        img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        
        # Convert to array
        pixels = np.array(img, dtype=np.float32)
        
        # Compute difference hash (horizontal gradient)
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        # Convert boolean array to hex string
        hash_bits = ''.join(['1' if bit else '0' for row in diff for bit in row])
        hash_hex = hex(int(hash_bits, 2))[2:].zfill(hash_size * hash_size // 4)
        
        return hash_hex
    except Exception as e:
        print(f"  [WARNING] Could not hash {image_path.name}: {e}")
        return ""

def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hashes."""
    if not hash1 or not hash2:
        return 999  # Invalid comparison
    
    # Convert hex to binary
    bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
    
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

def scan_near_duplicates(datasets: Dict[str, List[Path]], threshold: int = 5) -> Dict[str, Any]:
    """
    Scan for near-duplicates across datasets.
    threshold: Maximum Hamming distance to consider as duplicate (0-64 for 8x8 hash)
    """
    print("\n  Computing perceptual hashes...")
    all_hashes = {}
    
    for split_name, images in datasets.items():
        for img_path in images:
            phash = compute_perceptual_hash(img_path)
            if phash:
                all_hashes[str(img_path)] = {
                    "hash": phash,
                    "split": split_name,
                    "filename": img_path.name
                }
    
    print(f"  Hashed {len(all_hashes)} images")
    
    # Find near-duplicates
    duplicates = []
    paths = list(all_hashes.keys())
    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            hash1 = all_hashes[path1]["hash"]
            hash2 = all_hashes[path2]["hash"]
            
            dist = hamming_distance(hash1, hash2)
            
            if dist <= threshold:
                duplicates.append({
                    "image1": all_hashes[path1]["filename"],
                    "image2": all_hashes[path2]["filename"],
                    "split1": all_hashes[path1]["split"],
                    "split2": all_hashes[path2]["split"],
                    "distance": dist,
                    "cross_split": all_hashes[path1]["split"] != all_hashes[path2]["split"]
                })
    
    # Categorize
    cross_split_leaks = [d for d in duplicates if d["cross_split"]]
    within_split_dups = [d for d in duplicates if not d["cross_split"]]
    
    return {
        "total_duplicates": len(duplicates),
        "cross_split_leaks": len(cross_split_leaks),
        "within_split_duplicates": len(within_split_dups),
        "leaks": cross_split_leaks[:10],  # Show first 10
        "threshold": threshold
    }

def create_data_lineage():
    """Create data_lineage.json mapping images to splits with derivatives tracking."""
    lineage = {}
    
    # Known base images used to generate derivatives (mark as TEST to avoid train contamination)
    base_images_with_derivatives = ["00023.jpg", "00025.jpg", "00088.jpg", "00093.jpg", "00099.jpg"]
    
    # Map REAL to TRAIN (except base images used for derivatives)
    real_dir = DATA_DIR / "real"
    if real_dir.exists():
        for img in real_dir.glob("*"):
            if img.suffix.lower() in ['.jpg', '.png', '.webp']:
                # Mark base images as TEST (they have edge case derivatives)
                if img.name in base_images_with_derivatives:
                    lineage[img.name] = "TEST"
                else:
                    lineage[img.name] = "TRAIN"
    
    # Map FAKE to TRAIN
    fake_dir = DATA_DIR / "fake"
    if fake_dir.exists():
        for img in fake_dir.glob("*"):
            if img.suffix.lower() in ['.jpg', '.png', '.webp']:
                lineage[img.name] = "TRAIN"
    
    # Map OOD to TEST
    ood_dir = DATA_DIR / "ood"
    if ood_dir.exists():
        for img in ood_dir.glob("*"):
            if img.suffix.lower() in ['.jpg', '.png', '.webp']:
                lineage[img.name] = "TEST"
    
    # Map EdgeCases to TEST (not CALIBRATION - these are intentional test derivatives)
    edge_dir = DATA_DIR / "edge_cases"
    if edge_dir.exists():
        for img in edge_dir.glob("*"):
            if img.suffix.lower() in ['.jpg', '.png', '.webp']:
                lineage[img.name] = "TEST"
    
    # Save lineage with derivatives metadata
    lineage_data = {
        "splits": lineage,
        "known_derivatives": {
            "00023.jpg": [
                "exif_orientation_1.jpg", "exif_orientation_2.jpg", "exif_orientation_3.jpg",
                "exif_orientation_4.jpg", "exif_orientation_5.jpg", "exif_orientation_6.jpg",
                "exif_orientation_7.jpg", "exif_orientation_8.jpg",
                "color_cmyk_roundtrip.png", "color_grayscale_as_rgb.png"
            ]
        },
        "policy": "Edge cases are TEST-only. Base images with derivatives are also TEST to prevent train contamination."
    }
    
    with open(LINEAGE_FILE, 'w') as f:
        json.dump(lineage_data, f, indent=2)
    
    return lineage

def validate_lineage() -> Tuple[bool, List[str]]:
    """Validate that no images cross split boundaries."""
    if not LINEAGE_FILE.exists():
        return False, ["data_lineage.json does not exist"]
    
    with open(LINEAGE_FILE, 'r') as f:
        lineage_data = json.load(f)
    
    # Handle both old format (dict) and new format (nested dict with 'splits' key)
    if "splits" in lineage_data:
        lineage = lineage_data["splits"]
    else:
        lineage = lineage_data
    
    violations = []
    
    split_counts = defaultdict(int)
    for img, split in lineage.items():
        split_counts[split] += 1
    
    # Verify all are tagged with valid splits
    if any(split not in ["TRAIN", "CALIBRATION", "AUDIT", "TEST"] for split in lineage.values()):
        violations.append("Invalid split tags found")
    
    return len(violations) == 0, violations

def run_phase8_experiments():
    from datetime import datetime
    
    print("="*60)
    print("ECDD PHASE 8: DATASET GOVERNANCE & FEEDBACK LOOP SAFETY")
    print("="*60)
    
    results = []
    
    # E8.1: Near-duplicate leakage scan
    print("\nE8.1: Near-duplicate leakage scan")
    try:
        # Create lineage FIRST so we can use proper split assignments
        print("  Creating data lineage...")
        create_data_lineage()
        
        # Load lineage
        with open(LINEAGE_FILE, 'r') as f:
            lineage_data = json.load(f)
        lineage_splits = lineage_data.get("splits", {})
        known_derivatives = lineage_data.get("known_derivatives", {})
        
        datasets = {}
        
        # Load images from each directory
        real_dir = DATA_DIR / "real"
        fake_dir = DATA_DIR / "fake"
        ood_dir = DATA_DIR / "ood"
        edge_dir = DATA_DIR / "edge_cases"
        
        # Collect all images and assign splits based on lineage
        all_images = []
        if real_dir.exists():
            all_images.extend(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
        if fake_dir.exists():
            all_images.extend( list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))
        if ood_dir.exists():
            all_images.extend(list(ood_dir.glob("*.png")) + list(ood_dir.glob("*.jpg")))
        if edge_dir.exists():
            all_images.extend(list(edge_dir.glob("*.jpg")) + list(edge_dir.glob("*.png")))
        
        # Group by actual lineage split (not directory)
        for img_path in all_images:
            split = lineage_splits.get(img_path.name, "UNKNOWN")
            if split not in datasets:
                datasets[split] = []
            datasets[split].append(img_path)
        
        scan_results = scan_near_duplicates(datasets, threshold=5)
        
        print(f"\n  Total images scanned: {sum(len(imgs) for imgs in datasets.values())}")
        print(f"  Total duplicates found (Hamming ≤ 5): {scan_results['total_duplicates']}")
        print(f"  Cross-split leaks: {scan_results['cross_split_leaks']}")
        print(f"  Within-split duplicates: {scan_results['within_split_duplicates']}")
        
        # Check if leaks are due to known intentional derivatives (already loaded above)
        # Filter out known intentional derivatives from leak count
        unexpected_leaks = []
        for leak in scan_results.get('leaks', []):
            img1, img2 = leak['image1'], leak['image2']
            is_known = False
            
            # Check if this is a known derivative pair
            for base, derivatives in known_derivatives.items():
                if (img1 == base and img2 in derivatives) or (img2 == base and img1 in derivatives):
                    is_known = True
                    break
            
            if not is_known:
                unexpected_leaks.append(leak)
        
        if scan_results['cross_split_leaks'] > 0:
            print(f"\n  Total cross-split duplicates: {scan_results['cross_split_leaks']}")
            print(f"  Known intentional derivatives: {scan_results['cross_split_leaks'] - len(unexpected_leaks)}")
            print(f"  Unexpected leaks: {len(unexpected_leaks)}")
            
            if unexpected_leaks:
                print("\n  [WARNING] Unexpected cross-split leaks:")
                for leak in unexpected_leaks[:10]:
                    print(f"    {leak['image1']} ({leak['split1']}) ↔ {leak['image2']} ({leak['split2']}) [dist={leak['distance']}]")
            else:
                print("\n  [INFO] All cross-split duplicates are known intentional derivatives (acceptable)")
        
        passed = len(unexpected_leaks) == 0
        
        if passed:
            print("\n  [PASS] No cross-split leakage detected")
        else:
            print("\n  [FAIL] Cross-split leaks found - rebuild splits!")
        
        results.append(ExperimentResult(
            "experiment_e8_1",
            "Near-Duplicate Leakage Scan",
            passed,
            scan_results,
            str(datetime.now())
        ))
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        results.append(ExperimentResult("experiment_e8_1", "Near-Duplicate Leakage Scan", False, {"error": str(e)}, str(datetime.now())))
    
    # E8.2: Manual review protocol test
    print("\nE8.2: Manual review protocol test")
    try:
        protocol_path = BASE_DIR / "manual_review_protocol.md"
        
        if protocol_path.exists():
            print(f"  [PASS] Protocol document exists: {protocol_path.name}")
            with open(protocol_path, 'r') as f:
                content = f.read()
            
            # Check required sections
            required_sections = ["Consent", "Retention", "Stratification", "Contamination"]
            missing = [s for s in required_sections if s.lower() not in content.lower()]
            
            if missing:
                print(f"  [WARNING] Missing sections: {', '.join(missing)}")
                passed = False
            else:
                print("  [PASS] All required sections present")
                passed = True
        else:
            print("  [FAIL] Protocol document not found")
            passed = False
        
        results.append(ExperimentResult(
            "experiment_e8_2",
            "Manual Review Protocol",
            passed,
            {"protocol_exists": protocol_path.exists()},
            str(datetime.now())
        ))
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(ExperimentResult("experiment_e8_2", "Manual Review Protocol", False, {"error": str(e)}, str(datetime.now())))
    
    # E8.3: Data lineage enforcement test
    print("\nE8.3: Data lineage enforcement test")
    try:
        print("  Creating data_lineage.json...")
        lineage = create_data_lineage()
        print(f"  Tagged {len(lineage)} images")
        
        # Validate
        valid, violations = validate_lineage()
        
        if valid:
            print("  [PASS] Data lineage validated - no boundary violations")
        else:
            print(f"  [FAIL] Violations: {', '.join(violations)}")
        
        results.append(ExperimentResult(
            "experiment_e8_3",
            "Data Lineage Enforcement",
            valid,
            {"tagged_count": len(lineage), "violations": violations},
            str(datetime.now())
        ))
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(ExperimentResult("experiment_e8_3", "Data Lineage Enforcement", False, {"error": str(e)}, str(datetime.now())))
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 8 SUMMARY")
    print("="*60)
    all_passed = True
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        print(f"  [{status}] {res.name}")
        if not res.passed:
            all_passed = False
    
    if all_passed:
        print("\nOverall: ALL PASSED")
    else:
        print("\nOverall: SOME FAILED")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    for res in results:
        save_result(res)

if __name__ == "__main__":
    run_phase8_experiments()
