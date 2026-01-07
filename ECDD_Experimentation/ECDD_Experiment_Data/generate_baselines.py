"""
Generate Baseline Checkpoints for Golden Dataset
=================================================

Processes all images in the golden dataset and creates baseline checkpoints
for stages S0-S4 (stages S5-S8 require model inference and face detection).
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# Import checkpoint system
from checkpoint_system import (
    CheckpointStore, StageCheckpoint,
    compute_s0, compute_s1, compute_s3, compute_s4
)

# Paths - Updated for new folder structure
BASE_DIR = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation")
DATA_DIR = BASE_DIR / "ECDD_Experiment_Data"
POLICY_PATH = BASE_DIR / "policy_contract.yaml"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Dataset splits
SPLITS = {
    "real": DATA_DIR / "real",
    "fake": DATA_DIR / "fake", 
    "ood": DATA_DIR / "ood",
    "edge_cases": DATA_DIR / "edge_cases",
}

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def load_policy() -> dict:
    """Load policy contract from YAML file."""
    if POLICY_PATH.exists():
        with open(POLICY_PATH, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Fallback defaults
        return {
            "pixel_pipeline": {
                "exif": {"apply_orientation": True},
                "alpha": {"policy": "composite", "composite_background": [255, 255, 255]},
                "resize": {"target_size": [256, 256], "interpolation": "bilinear", "antialias": True},
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            }
        }


def convert_policy_format(raw_policy: dict) -> dict:
    """Convert YAML policy format to internal format."""
    pp = raw_policy.get('pixel_pipeline', raw_policy)
    return {
        "exif": pp.get('exif', {}),
        "alpha": pp.get('alpha', {}),
        "resize": pp.get('resize', {}),
        "normalization": pp.get('normalization', {}),
    }


def generate_baselines():
    """Generate baseline checkpoints for all golden dataset images."""
    print("=" * 60)
    print("ECDD Baseline Checkpoint Generation")
    print("=" * 60)
    
    # Load policy
    raw_policy = load_policy()
    policy = convert_policy_format(raw_policy)
    print(f"Loaded policy (resize: {policy['resize'].get('target_size', [256, 256])})")
    
    # Initialize checkpoint store
    store = CheckpointStore(CHECKPOINT_DIR)
    
    # Track stats
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "by_split": {}
    }
    
    # Process each split
    for split_name, split_dir in SPLITS.items():
        if not split_dir.exists():
            print(f"\n[SKIP] {split_name}: directory not found")
            continue
        
        # Find all images
        images = [
            f for f in split_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        
        print(f"\n[{split_name.upper()}] Processing {len(images)} images...")
        stats["by_split"][split_name] = {"total": len(images), "processed": 0, "errors": 0}
        
        for img_path in images:
            try:
                # Skip if already processed
                existing = store.load_checkpoint(img_path.stem)
                if existing and existing.s0 and existing.s4:
                    stats["skipped"] += 1
                    continue
                
                # Compute stages
                s0 = compute_s0(img_path)
                s1 = compute_s1(img_path, policy)
                s3, s3_tensor = compute_s3(img_path, policy)
                s4, _ = compute_s4(s3_tensor, policy)
                
                # Create checkpoint
                checkpoint = StageCheckpoint(
                    image_path=str(img_path),
                    image_id=img_path.stem,
                    dataset_split=split_name,
                    created_at=datetime.now().isoformat(),
                    pipeline_version=raw_policy.get('version', '1.0'),
                    s0=s0,
                    s1=s1,
                    s3=s3,
                    s4=s4,
                )
                store.save_checkpoint(checkpoint)
                
                stats["processed"] += 1
                stats["by_split"][split_name]["processed"] += 1
                print(f"  [OK] {img_path.name}")
                
            except Exception as e:
                stats["errors"] += 1
                stats["by_split"][split_name]["errors"] += 1
                print(f"  [ERROR] {img_path.name}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Processed: {stats['processed']}")
    print(f"Skipped (already exists): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nBy split:")
    for split_name, split_stats in stats['by_split'].items():
        print(f"  {split_name}: {split_stats['processed']}/{split_stats['total']} "
              f"(errors: {split_stats['errors']})")
    
    # Print store summary
    summary = store.get_summary()
    print(f"\nCheckpoint store summary:")
    print(f"  Total checkpoints: {summary['total_checkpoints']}")
    print(f"  Stage coverage: S0={summary['stage_coverage']['s0']}, "
          f"S1={summary['stage_coverage']['s1']}, "
          f"S3={summary['stage_coverage']['s3']}, "
          f"S4={summary['stage_coverage']['s4']}")


if __name__ == "__main__":
    generate_baselines()
