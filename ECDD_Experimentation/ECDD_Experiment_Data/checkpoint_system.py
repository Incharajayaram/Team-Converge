"""
ECDD Stage Checkpoint System
============================

This module provides the infrastructure for computing and storing stage checkpoints (S0-S8)
for the ECDD pipeline as defined in the experimentation document.

Stage Definitions:
- S0: Raw bytes hash (at browser and server ingress)
- S1: Decoded RGB tensor hash (post-EXIF, post-alpha, post-gamma)
- S2: Face crop boxes and aligned crop tensors hash
- S3: Resized 256x256 tensor hash
- S4: Normalized tensor hash
- S5: Patch-logit map stats hash (shape, min/max/mean)
- S6: Pooled logit (float)
- S7: Calibrated logit and calibrated probability
- S8: Decision label and reason codes
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')


# ============================================================================
# Data Classes for Stage Checkpoints
# ============================================================================

@dataclass
class S0_RawBytes:
    """Stage 0: Raw bytes hash at ingress."""
    sha256: str
    file_size_bytes: int
    file_extension: str


@dataclass 
class S1_DecodedRGB:
    """Stage 1: Decoded RGB tensor hash (post-EXIF, post-alpha, post-gamma)."""
    tensor_hash: str
    shape: Tuple[int, int, int]  # (H, W, C)
    dtype: str
    min_val: float
    max_val: float
    mean_val: float


@dataclass
class S2_FaceCrops:
    """Stage 2: Face crop boxes and aligned crop tensors."""
    num_faces: int
    boxes: List[Dict[str, float]]  # List of {x1, y1, x2, y2, confidence}
    crop_hashes: List[str]  # Hash of each aligned crop tensor
    detector_model: str
    detector_version: str


@dataclass
class S3_ResizedTensor:
    """Stage 3: Resized 256x256 tensor hash."""
    tensor_hash: str
    shape: Tuple[int, int, int]
    interpolation_method: str
    min_val: float
    max_val: float
    mean_val: float


@dataclass
class S4_NormalizedTensor:
    """Stage 4: Normalized tensor hash."""
    tensor_hash: str
    shape: Tuple[int, int, int]
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    normalization_mean: List[float]
    normalization_std: List[float]


@dataclass
class S5_PatchLogitMap:
    """Stage 5: Patch-logit map stats."""
    shape: Tuple[int, int]  # (H, W) of patch grid
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    stats_hash: str  # Hash of (shape, min, max, mean) for quick comparison


@dataclass
class S6_PooledLogit:
    """Stage 6: Pooled logit (single float)."""
    logit_value: float
    pooling_method: str
    pooling_params: Dict[str, Any]


@dataclass
class S7_CalibratedOutput:
    """Stage 7: Calibrated logit and probability."""
    calibrated_logit: float
    calibrated_probability: float
    calibration_method: str
    calibration_params: Dict[str, float]


@dataclass
class S8_Decision:
    """Stage 8: Decision label and reason codes."""
    label: str  # "Real", "Fake", or "Abstain"
    probability: float
    confidence_level: str  # "high", "medium", "low"
    reason_codes: List[str]  # e.g., ["GR-001"] for no-face abstain
    threshold_used: float


@dataclass
class StageCheckpoint:
    """Complete checkpoint for a single image."""
    image_path: str
    image_id: str
    dataset_split: str  # "real", "fake", "ood", "edge_cases"
    created_at: str
    pipeline_version: str
    
    s0: Optional[S0_RawBytes] = None
    s1: Optional[S1_DecodedRGB] = None
    s2: Optional[S2_FaceCrops] = None
    s3: Optional[S3_ResizedTensor] = None
    s4: Optional[S4_NormalizedTensor] = None
    s5: Optional[S5_PatchLogitMap] = None
    s6: Optional[S6_PooledLogit] = None
    s7: Optional[S7_CalibratedOutput] = None
    s8: Optional[S8_Decision] = None


# ============================================================================
# Hashing Utilities
# ============================================================================

def compute_bytes_hash(data: bytes) -> str:
    """Compute SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_tensor_hash(tensor: np.ndarray) -> str:
    """
    Compute a deterministic hash of a numpy tensor.
    Uses the raw bytes of the array for exact reproducibility.
    """
    # Ensure contiguous C-order array
    tensor_c = np.ascontiguousarray(tensor)
    # Create hash from raw bytes
    return hashlib.sha256(tensor_c.tobytes()).hexdigest()


def compute_stats_hash(shape: tuple, min_val: float, max_val: float, mean_val: float) -> str:
    """Compute hash of summary statistics for quick comparison."""
    stats_str = f"{shape}|{min_val:.6f}|{max_val:.6f}|{mean_val:.6f}"
    return hashlib.sha256(stats_str.encode()).hexdigest()[:16]


# ============================================================================
# Stage Computation Functions
# ============================================================================

def compute_s0(file_path: Path) -> S0_RawBytes:
    """Compute Stage 0: Raw bytes hash."""
    raw_bytes = file_path.read_bytes()
    return S0_RawBytes(
        sha256=compute_bytes_hash(raw_bytes),
        file_size_bytes=len(raw_bytes),
        file_extension=file_path.suffix.lower()
    )


def compute_s1(file_path: Path, policy: dict) -> S1_DecodedRGB:
    """
    Compute Stage 1: Decoded RGB tensor.
    
    Applies:
    - EXIF orientation correction
    - Alpha channel handling (composite or reject)
    - Color space standardization (sRGB)
    - RGB channel ordering enforcement
    """
    from PIL import Image, ImageOps
    
    # Open image
    img = Image.open(file_path)
    
    # Apply EXIF orientation if configured
    if policy.get('exif', {}).get('apply_orientation', True):
        img = ImageOps.exif_transpose(img)
    
    # Handle alpha channel
    alpha_policy = policy.get('alpha', {}).get('policy', 'composite')
    if img.mode == 'RGBA':
        if alpha_policy == 'composite':
            bg_color = tuple(policy.get('alpha', {}).get('composite_background', [255, 255, 255]))
            background = Image.new('RGB', img.size, bg_color)
            background.paste(img, mask=img.split()[3])
            img = background
        elif alpha_policy == 'reject':
            raise ValueError("Image has alpha channel and policy is 'reject'")
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    tensor = np.array(img, dtype=np.uint8)
    
    return S1_DecodedRGB(
        tensor_hash=compute_tensor_hash(tensor),
        shape=tensor.shape,
        dtype=str(tensor.dtype),
        min_val=float(tensor.min()),
        max_val=float(tensor.max()),
        mean_val=float(tensor.mean())
    )


def compute_s3(s1_tensor_or_path, policy: dict) -> Tuple[S3_ResizedTensor, np.ndarray]:
    """
    Compute Stage 3: Resized 256x256 tensor.
    
    Note: S2 (face detection) is skipped for now as it requires a face detector.
    This function takes the decoded tensor and resizes it.
    """
    from PIL import Image, ImageOps
    
    # If we're given a path, decode first (simplified for testing without S2)
    if isinstance(s1_tensor_or_path, (str, Path)):
        img = Image.open(s1_tensor_or_path)
        img = ImageOps.exif_transpose(img)
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        img = Image.fromarray(s1_tensor_or_path)
    
    # Get resize parameters from policy
    target_size = tuple(policy.get('resize', {}).get('target_size', [256, 256]))
    interp_name = policy.get('resize', {}).get('interpolation', 'bilinear')
    antialias = policy.get('resize', {}).get('antialias', True)
    
    # Map interpolation name to PIL constant
    interp_map = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
    }
    interp_method = interp_map.get(interp_name, Image.Resampling.BILINEAR)
    
    # Resize
    img_resized = img.resize((target_size[1], target_size[0]), interp_method)
    tensor = np.array(img_resized, dtype=np.uint8)
    
    return S3_ResizedTensor(
        tensor_hash=compute_tensor_hash(tensor),
        shape=tensor.shape,
        interpolation_method=interp_name,
        min_val=float(tensor.min()),
        max_val=float(tensor.max()),
        mean_val=float(tensor.mean())
    ), tensor


def compute_s4(s3_tensor: np.ndarray, policy: dict) -> Tuple[S4_NormalizedTensor, np.ndarray]:
    """
    Compute Stage 4: Normalized tensor.
    
    Applies mean/std normalization per channel.
    """
    # Get normalization parameters from policy
    mean = np.array(policy.get('normalization', {}).get('mean', [0.485, 0.456, 0.406]))
    std = np.array(policy.get('normalization', {}).get('std', [0.229, 0.224, 0.225]))
    
    # Convert to float and scale to 0-1
    tensor = s3_tensor.astype(np.float32) / 255.0
    
    # Apply normalization: (x - mean) / std
    tensor = (tensor - mean) / std
    
    return S4_NormalizedTensor(
        tensor_hash=compute_tensor_hash(tensor.astype(np.float32)),
        shape=tensor.shape,
        dtype='float32',
        min_val=float(tensor.min()),
        max_val=float(tensor.max()),
        mean_val=float(tensor.mean()),
        std_val=float(tensor.std()),
        normalization_mean=mean.tolist(),
        normalization_std=std.tolist()
    ), tensor


# ============================================================================
# Checkpoint Storage
# ============================================================================

class CheckpointStore:
    """Manages storage and retrieval of stage checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.checkpoint_dir / "checkpoint_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load the checkpoint index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "checkpoints": {}
            }
    
    def _save_index(self):
        """Save the checkpoint index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _checkpoint_path(self, image_id: str) -> Path:
        """Get the path for a checkpoint file."""
        return self.checkpoint_dir / f"{image_id}.json"
    
    def save_checkpoint(self, checkpoint: StageCheckpoint):
        """Save a checkpoint to disk."""
        # Convert dataclass to dict, handling nested dataclasses
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [to_dict(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        checkpoint_dict = to_dict(checkpoint)
        
        # Save checkpoint file
        checkpoint_path = self._checkpoint_path(checkpoint.image_id)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_dict, f, indent=2)
        
        # Update index
        self.index["checkpoints"][checkpoint.image_id] = {
            "path": str(checkpoint_path.relative_to(self.checkpoint_dir)),
            "image_path": checkpoint.image_path,
            "dataset_split": checkpoint.dataset_split,
            "created_at": checkpoint.created_at,
            "has_stages": {
                f"s{i}": getattr(checkpoint, f's{i}') is not None
                for i in range(9)
            }
        }
        self._save_index()
    
    def load_checkpoint(self, image_id: str) -> Optional[StageCheckpoint]:
        """Load a checkpoint from disk."""
        checkpoint_path = self._checkpoint_path(image_id)
        if not checkpoint_path.exists():
            return None
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct dataclasses
        def from_dict(cls, data):
            if data is None:
                return None
            if hasattr(cls, '__dataclass_fields__'):
                fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
                return cls(**{k: from_dict(fieldtypes.get(k, type(v)), v) for k, v in data.items()})
            return data
        
        # Reconstruct each stage
        checkpoint = StageCheckpoint(
            image_path=data['image_path'],
            image_id=data['image_id'],
            dataset_split=data['dataset_split'],
            created_at=data['created_at'],
            pipeline_version=data['pipeline_version'],
            s0=S0_RawBytes(**data['s0']) if data.get('s0') else None,
            s1=S1_DecodedRGB(**data['s1']) if data.get('s1') else None,
            s2=S2_FaceCrops(**data['s2']) if data.get('s2') else None,
            s3=S3_ResizedTensor(**data['s3']) if data.get('s3') else None,
            s4=S4_NormalizedTensor(**data['s4']) if data.get('s4') else None,
            s5=S5_PatchLogitMap(**data['s5']) if data.get('s5') else None,
            s6=S6_PooledLogit(**data['s6']) if data.get('s6') else None,
            s7=S7_CalibratedOutput(**data['s7']) if data.get('s7') else None,
            s8=S8_Decision(**data['s8']) if data.get('s8') else None,
        )
        return checkpoint
    
    def list_checkpoints(self, dataset_split: Optional[str] = None) -> List[str]:
        """List all checkpoint image IDs, optionally filtered by split."""
        checkpoints = self.index.get("checkpoints", {})
        if dataset_split:
            return [k for k, v in checkpoints.items() if v.get("dataset_split") == dataset_split]
        return list(checkpoints.keys())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all checkpoints."""
        summary = {
            "total_checkpoints": len(self.index.get("checkpoints", {})),
            "by_split": {},
            "stage_coverage": {f"s{i}": 0 for i in range(9)}
        }
        
        for image_id, meta in self.index.get("checkpoints", {}).items():
            split = meta.get("dataset_split", "unknown")
            summary["by_split"][split] = summary["by_split"].get(split, 0) + 1
            
            for stage, has_it in meta.get("has_stages", {}).items():
                if has_it:
                    summary["stage_coverage"][stage] += 1
        
        return summary


# ============================================================================
# Comparison Utilities
# ============================================================================

def compare_checkpoints(golden: StageCheckpoint, current: StageCheckpoint, 
                       tolerances: dict) -> Dict[str, Any]:
    """
    Compare two checkpoints and report differences.
    
    Returns a dict with:
    - passed: bool indicating if all stages match within tolerances
    - stages: dict of stage results with details
    """
    results = {"passed": True, "stages": {}}
    
    # S0: Exact byte match required
    if golden.s0 and current.s0:
        match = golden.s0.sha256 == current.s0.sha256
        results["stages"]["s0"] = {
            "passed": match,
            "golden_hash": golden.s0.sha256[:16] + "...",
            "current_hash": current.s0.sha256[:16] + "...",
        }
        if not match:
            results["passed"] = False
    
    # S1: Decoded tensor - check hash or allow tolerance
    if golden.s1 and current.s1:
        tol = tolerances.get('s1_decoded_max_diff', 0)
        if tol == 0:
            match = golden.s1.tensor_hash == current.s1.tensor_hash
        else:
            # For non-zero tolerance, compare stats
            match = abs(golden.s1.mean_val - current.s1.mean_val) <= tol
        results["stages"]["s1"] = {
            "passed": match,
            "shape_match": golden.s1.shape == current.s1.shape,
        }
        if not match:
            results["passed"] = False
    
    # S3: Resized tensor
    if golden.s3 and current.s3:
        tol = tolerances.get('s3_resized_max_diff', 1)
        mean_diff = abs(golden.s3.mean_val - current.s3.mean_val)
        match = mean_diff <= tol and golden.s3.shape == current.s3.shape
        results["stages"]["s3"] = {
            "passed": match,
            "shape_match": golden.s3.shape == current.s3.shape,
            "mean_diff": mean_diff,
        }
        if not match:
            results["passed"] = False
    
    # S4: Normalized tensor
    if golden.s4 and current.s4:
        tol = tolerances.get('s4_normalized_max_diff', 1e-5)
        mean_diff = abs(golden.s4.mean_val - current.s4.mean_val)
        match = mean_diff <= tol
        results["stages"]["s4"] = {
            "passed": match,
            "mean_diff": mean_diff,
        }
        if not match:
            results["passed"] = False
    
    return results


# ============================================================================
# Main Entry Point for Testing
# ============================================================================

if __name__ == "__main__":
    print("ECDD Stage Checkpoint System")
    print("=" * 50)
    
    # Test basic functionality
    from PIL import ImageOps
    
    DATA_DIR = Path(r"f:\Team converge\Team-Converge\Experimentation\ECDD_Experiment_Data")
    CHECKPOINT_DIR = DATA_DIR / "checkpoints"
    
    # Initialize store
    store = CheckpointStore(CHECKPOINT_DIR)
    
    # Load policy from YAML (simplified inline for testing)
    policy = {
        "exif": {"apply_orientation": True},
        "alpha": {"policy": "composite", "composite_background": [255, 255, 255]},
        "resize": {"target_size": [256, 256], "interpolation": "bilinear", "antialias": True},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    
    # Process a few test images
    test_images = list(DATA_DIR.glob("real/*.jpg"))[:3]
    
    for img_path in test_images:
        print(f"\nProcessing: {img_path.name}")
        
        # Compute stages
        s0 = compute_s0(img_path)
        print(f"  S0: {s0.sha256[:16]}... ({s0.file_size_bytes} bytes)")
        
        s1 = compute_s1(img_path, policy)
        print(f"  S1: shape={s1.shape}, mean={s1.mean_val:.2f}")
        
        s3, s3_tensor = compute_s3(img_path, policy)
        print(f"  S3: shape={s3.shape}, interp={s3.interpolation_method}")
        
        s4, s4_tensor = compute_s4(s3_tensor, policy)
        print(f"  S4: mean={s4.mean_val:.4f}, std={s4.std_val:.4f}")
        
        # Create and save checkpoint
        checkpoint = StageCheckpoint(
            image_path=str(img_path),
            image_id=img_path.stem,
            dataset_split="real",
            created_at=datetime.now().isoformat(),
            pipeline_version="1.0",
            s0=s0,
            s1=s1,
            s3=s3,
            s4=s4,
        )
        store.save_checkpoint(checkpoint)
        print(f"  [OK] Saved checkpoint")
    
    # Print summary
    print("\n" + "=" * 50)
    summary = store.get_summary()
    print(f"Total checkpoints: {summary['total_checkpoints']}")
    print(f"By split: {summary['by_split']}")
    print(f"Stage coverage: {summary['stage_coverage']}")
