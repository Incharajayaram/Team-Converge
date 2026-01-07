"""
ECDD Dataset Preparation and Management
Downloads, organizes, and prepares datasets for training ECDD models.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import shutil
import random
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: opencv-python not installed. Video extraction will fail.")
    cv2 = None
    CV2_AVAILABLE = False

# Configuration
BASE_DIR = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation")
DATASET_DIR = BASE_DIR / "ECDD_Training_Data"
DOWNLOAD_DIR = DATASET_DIR / "downloads"
PROCESSED_DIR = DATASET_DIR / "processed"

# Dataset URLs and info
DATASET_REGISTRY = {
    "deepfake_eval_2024": {
        "name": "Deepfake-Eval-2024",
        "type": "huggingface",
        "url": "https://huggingface.co/datasets/TrueMedia/Deepfake-Eval-2024",
        "description": "In-the-wild deepfakes from 2024 (TrueMedia.org)",
        "size_gb": 50,
        "priority": "HIGH"
    },
    "realguard_2025": {
        "name": "RealGuard-2025",
        "type": "kaggle",
        "url": "https://www.kaggle.com/datasets/realguard-2025",
        "description": "Sora, RunwayML, Pika Labs, HeyGen content",
        "size_gb": 2,
        "priority": "HIGH"
    },
    "ff_plus_plus": {
        "name": "FaceForensics++",
        "type": "academic",
        "url": "https://github.com/ondyari/FaceForensics",
        "description": "Classic benchmark: Face2Face, FaceSwap, Deepfakes, NT",
        "size_gb": 100,
        "priority": "MEDIUM"
    },
    "celeb_df_v2": {
        "name": "Celeb-DF v2",
        "type": "academic",
        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "description": "High-quality celebrity face swaps",
        "size_gb": 10,
        "priority": "MEDIUM"
    }
}

@dataclass
class DatasetSplit:
    """Defines a dataset split configuration."""
    name: str
    purpose: str
    generators: List[str]
    min_size: int
    max_size: Optional[int] = None

# Split configuration per implementation plan
SPLIT_CONFIG = {
    "TRAIN": DatasetSplit(
        name="TRAIN",
        purpose="Base model training",
        generators=["ff_f2f", "ff_faceswap", "ff_deepfakes", "celeb_df"],
        min_size=5000
    ),
    "FINETUNE": DatasetSplit(
        name="FINETUNE",
        purpose="Modern generator adaptation",
        generators=["deepfake_eval_2024", "self_generated"],
        min_size=1000
    ),
    "CALIBRATION": DatasetSplit(
        name="CALIBRATION",
        purpose="Threshold fitting (deployment-like)",
        generators=["phone_captures", "web_uploads"],
        min_size=500
    ),
    "TEST_SEEN": DatasetSplit(
        name="TEST_SEEN",
        purpose="Held-out same family",
        generators=["ff_neuraltextures"],
        min_size=500
    ),
    "TEST_UNSEEN": DatasetSplit(
        name="TEST_UNSEEN",
        purpose="Generalization testing",
        generators=["realguard_2025", "new_generators"],
        min_size=200
    )
}


def create_directory_structure():
    """Create the dataset directory structure."""
    dirs = [
        DATASET_DIR,
        DOWNLOAD_DIR,
        PROCESSED_DIR,
        PROCESSED_DIR / "real",
        PROCESSED_DIR / "fake",
        PROCESSED_DIR / "splits" / "train",
        PROCESSED_DIR / "splits" / "finetune",
        PROCESSED_DIR / "splits" / "calibration",
        PROCESSED_DIR / "splits" / "test_seen",
        PROCESSED_DIR / "splits" / "test_unseen",
        DATASET_DIR / "metadata"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")


def generate_dataset_config():
    """Generate dataset configuration file."""
    config = {
        "created": datetime.now().isoformat(),
        "version": "1.0",
        "splits": {
            name: {
                "purpose": split.purpose,
                "generators": split.generators,
                "min_size": split.min_size,
                "max_size": split.max_size,
                "current_size": 0
            }
            for name, split in SPLIT_CONFIG.items()
        },
        "datasets": DATASET_REGISTRY,
        "preprocessing": {
            "target_size": [256, 256],
            "interpolation": "LANCZOS",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }
    
    config_path = DATASET_DIR / "dataset_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Config saved: {config_path}")
    return config


def check_existing_data():
    """Check what data is already available."""
    existing = {
        "current_experiment_data": 0,
        "processed_real": 0,
        "processed_fake": 0
    }
    
    # Check current experiment data
    current_data = BASE_DIR / "ECDD_Experiment_Data"
    if current_data.exists():
        for split in ["real", "fake"]:
            split_dir = current_data / split
            if split_dir.exists():
                count = len(list(split_dir.glob("*.[jp][pn][g]")))
                existing[f"current_{split}"] = count
                existing["current_experiment_data"] += count
    
    return existing


def extract_frames(video_path: Path, output_dir: Path, num_frames: int = 3) -> List[Path]:
    """Extract frames from video file."""
    if not cv2:
        return []
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
        
    extracted = []
    video_name = video_path.stem
    
    # Select equidistant frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out_name = f"{video_name}_frame{i}.jpg"
            out_path = output_dir / out_name
            # Only save if not exists (resume capability)
            if not out_path.exists():
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append(out_path)
            
    cap.release()
    return extracted


def find_sources_recursive(root_dir: Path, extensions: set) -> List[Path]:
    """Recursively find files with specific extensions."""
    found = []
    try:
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                found.append(p)
    except Exception as e:
        print(f"Error scanning {root_dir}: {e}")
    return found


def select_balanced_subset(images: List[Path], target_count: int) -> List[Path]:
    """Select a balanced subset of images."""
    if not images:
        return []
        
    # Shuffle to avoid alphabetical bias
    random.shuffle(images)
    
    if len(images) <= target_count:
        return images
    
    return images[:target_count]


def process_datasets(dry_run: bool = False):
    """Process datasets into split structure according to plan."""
    print("\nProcessing datasets for 10,000 image target...")
    
    DS_ROOT = DATASET_DIR / "ECDD_Datasets"
    print(f"  Looking for datasets in: {DS_ROOT}")
    
    EXTRACT_DIR = PROCESSED_DIR / "extracted"
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Target distribution
    targets = {
        "TRAIN": 5000,
        "FINETUNE": 1000,
        "CALIBRATION": 500,
        "TEST_SEEN": 500,
        "TEST_UNSEEN": 500
    }
    
    # 1. Map Source Videos/Images
    # We look for videos in known paths
    source_files = {
        "ff_train_vid": [],
        "ff_test_vid": [],
        "celeb_vid": [],
        "modern_train": [], # Assumed images? Or videos? New2/Train
        "modern_val": [],
        "modern_test": [],
        "realguard_img": []
    }
    
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    
    # FF++ (New1) - Contains IMAGES not videos!
    # Structure: New1/Data Set X/Data Set X/{train|test}/{fake|real}/*.jpg
    new1 = DS_ROOT / "New1"
    if new1.exists():
        print("  Scanning New1 (FF++ images)...")
        # Find all JPG files recursively
        for p in find_sources_recursive(new1, image_exts):
            p_str = str(p).lower()
            # Classify by train/test folder in path
            if "\\train\\" in p_str or "/train/" in p_str:
                source_files["ff_train_vid"].append(p)  # Reusing vid list but with images
            elif "\\test\\" in p_str or "/test/" in p_str:
                source_files["ff_test_vid"].append(p)

    # Celeb-DF
    celeb = DS_ROOT / "Celeb-DF-v2"
    if celeb.exists():
        print("  Scanning Celeb-DF...")
        source_files["celeb_vid"] = find_sources_recursive(celeb, video_exts)
        
    # Modern (New2) - Check if video or image
    new2 = DS_ROOT / "New2"
    if new2.exists():
        print("  Scanning New2 (Modern)...")
        # Assume mixed or check first file
        found_any = find_sources_recursive(new2, video_exts | image_exts)
        for p in found_any:
            p_str = str(p).lower()
            if "train" in p_str:
                source_files["modern_train"].append(p)
            elif "valid" in p_str:
                source_files["modern_val"].append(p)
            elif "test" in p_str:
                source_files["modern_test"].append(p)

    # RealGuard
    realguard = DS_ROOT / "realguard_2025"
    if realguard.exists():
        print("  Scanning RealGuard...")
        source_files["realguard_img"] = find_sources_recursive(realguard, image_exts)

    print(f"  Source counts (Files):")
    print(f"    FF++ Train Vid: {len(source_files['ff_train_vid'])}")
    print(f"    Celeb Vid: {len(source_files['celeb_vid'])}")
    print(f"    Modern Train Items: {len(source_files['modern_train'])}")

    # 2. Extract Frames Helper
    def prepare_pool(files: List[Path], pool_name: str, frames_per_vid: int = 3) -> List[Path]:
        """Convert list of files (videos/images) to a list of usable image paths."""
        images = []
        pool_dir = EXTRACT_DIR / pool_name
        pool_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"    Preparing {pool_name} (processing {len(files)} files)...")
        
        # Limit to 2000 files if we have too many
        files_to_process = files[:2000] if len(files) > 2000 else files
        
        for p in files_to_process:
            if p.suffix.lower() in video_exts:
                # Extract frames from video
                extracted = extract_frames(p, pool_dir, frames_per_vid)
                images.extend(extracted)
            elif p.suffix.lower() in image_exts:
                # It's already an image, use directly
                images.append(p)
                
        print(f"      -> Collected {len(images)} images")
        return images

    # 3. Populate Pools (Extract/Collect)
    print("\n  Extracting/Collecting Frames...")
    
    # FF++ Train Pool
    pool_ff_train = prepare_pool(source_files["ff_train_vid"], "ff_train", frames_per_vid=1)
    # Celeb Pool
    pool_celeb = prepare_pool(source_files["celeb_vid"], "celeb", frames_per_vid=1)
    
    # Modern Pools (Handle potential mixed content)
    pool_modern_train = prepare_pool(source_files["modern_train"], "modern_train", frames_per_vid=1)
    pool_modern_val = prepare_pool(source_files["modern_val"], "modern_val", frames_per_vid=1)
    pool_modern_test = prepare_pool(source_files["modern_test"], "modern_test", frames_per_vid=1)
    
    # Test Pools
    pool_ff_test = prepare_pool(source_files["ff_test_vid"], "ff_test", frames_per_vid=1)
    pool_realguard = prepare_pool(source_files["realguard_img"], "realguard")

    # 4. Copy to Final Splits
    def copy_to_split(image_list, split_name, limit):
        dst_dir = PROCESSED_DIR / "splits" / split_name.lower()
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        selected = select_balanced_subset(image_list, limit)
        count = 0
        
        for src in selected:
            if not dry_run:
                try:
                    # Name logic: preserve lineage
                    name = src.name
                    if "frame" not in name and src.parent.name != "splits":
                        name = f"{src.parent.name}_{name}"
                        
                    target = dst_dir / name
                    # De-duplicate name
                    if target.exists():
                        target = dst_dir / f"{count}_{name}"
                        
                    shutil.copy2(src, target)
                    count += 1
                except Exception as e:
                    print(f"    Failed copy: {e}")
        print(f"    -> {split_name}: {count}/{limit}")
        return count

    print("\n  Populating Final Splits...")
    
    # TRAIN
    ff_quota = int(targets["TRAIN"] * 0.7)
    copy_to_split(pool_ff_train, "train", ff_quota)
    copy_to_split(pool_celeb, "train", targets["TRAIN"] - ff_quota)
    
    # FINETUNE
    copy_to_split(pool_modern_train, "finetune", targets["FINETUNE"])
    
    # CALIBRATION
    copy_to_split(pool_modern_val, "calibration", targets["CALIBRATION"])
    
    # TEST
    copy_to_split(pool_ff_test, "test_seen", targets["TEST_SEEN"])
    copy_to_split(pool_modern_test, "test_unseen", 400)
    copy_to_split(pool_realguard, "test_unseen", 100)
    
    print("\n  Done!")


def main():
    print("="*60)
    print("ECDD DATASET PREPARATION")
    print("="*60)
    
    # 1. Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    # 2. Generate config
    print("\n2. Generating dataset configuration...")
    generate_dataset_config()
    
    # 3. Process
    print("\n3. Processing video/image datasets...")
    process_datasets(dry_run=False)
    
    print("\n" + "="*60)
    print("DONE")


if __name__ == "__main__":
    main()
