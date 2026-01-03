"""Celeb-DF v2 benchmark dataset loader for external evaluation.

Celeb-DF v2 is a large-scale challenging deepfake dataset specifically
designed to test model generalization.

Paper: https://arxiv.org/abs/1909.12962
Dataset: https://github.com/yuezunli/celeb-deepfakeforensics

Directory structure expected:
    celeb_df_root/
    ├── Celeb-real/        # Real celebrity videos
    ├── Celeb-synthesis/   # Deepfake videos
    └── YouTube-real/      # Real YouTube videos

Usage:
    This dataset should be used as TEST-ONLY for benchmark evaluation.
    Do NOT use for training or threshold calibration.
"""

import os
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import json


class CelebDFDataset(Dataset):
    """
    Celeb-DF v2 benchmark dataset for deepfake detection.
    
    Designed for test-only evaluation with identical preprocessing
    to internal datasets (resize 256, normalize ImageNet).
    
    Key features:
    - Video-level test-only split (no train/val contamination)
    - Frame extraction at configurable FPS (default: 1 fps)
    - Identical preprocessing to internal dataset
    - Caches extracted frames on disk for faster re-loading
    """
    
    def __init__(
        self,
        root_dir: str,
        fps: float = 1.0,
        max_frames_per_video: Optional[int] = None,
        resize_size: int = 256,
        normalize: bool = True,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
        cache_dir: Optional[str] = None,
        force_extract: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing Celeb-real/, Celeb-synthesis/, YouTube-real/
            fps: Frames per second to extract (default: 1.0)
            max_frames_per_video: Maximum frames to extract per video (None = all)
            resize_size: Target size for resizing (must match internal dataset: 256)
            normalize: Whether to normalize with ImageNet constants
            normalize_mean: Normalization mean (default: ImageNet)
            normalize_std: Normalization std (default: ImageNet)
            cache_dir: Directory to cache extracted frames (default: root_dir/frames_cache)
            force_extract: Force re-extraction even if cache exists
        """
        self.root_dir = Path(root_dir)
        self.fps = fps
        self.max_frames_per_video = max_frames_per_video
        self.resize_size = resize_size
        self.normalize = normalize
        self.force_extract = force_extract
        
        # Use ImageNet normalization by default (must match internal dataset)
        if normalize_mean is None:
            self.normalize_mean = np.array([0.485, 0.456, 0.406])
        else:
            self.normalize_mean = np.array(normalize_mean)
        
        if normalize_std is None:
            self.normalize_std = np.array([0.229, 0.224, 0.225])
        else:
            self.normalize_std = np.array(normalize_std)
        
        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = self.root_dir / "frames_cache" / f"fps_{fps}"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # Load or extract frames
        self.samples = []  # List of (frame_path, label, video_name) tuples
        self._load_or_extract_frames()
    
    def _load_or_extract_frames(self):
        """Load frames from cache or extract from videos."""
        if self.metadata_file.exists() and not self.force_extract:
            print(f"Loading frames from cache: {self.cache_dir}")
            self._load_from_cache()
        else:
            print(f"Extracting frames from videos (fps={self.fps})...")
            self._extract_frames_from_videos()
            self._save_metadata()
        
        print(f"✓ Loaded {len(self.samples)} frames from Celeb-DF v2")
        print(f"  Real frames: {sum(1 for _, label, _ in self.samples if label == 0)}")
        print(f"  Fake frames: {sum(1 for _, label, _ in self.samples if label == 1)}")
    
    def _load_from_cache(self):
        """Load frame paths from cached metadata."""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for entry in metadata['frames']:
            frame_path = self.cache_dir / entry['path']
            if frame_path.exists():
                self.samples.append((str(frame_path), entry['label'], entry['video']))
    
    def _extract_frames_from_videos(self):
        """Extract frames from Celeb-DF videos at specified FPS."""
        # Define directory structure
        video_dirs = [
            (self.root_dir / "Celeb-real", 0, "real"),
            (self.root_dir / "YouTube-real", 0, "real"),
            (self.root_dir / "Celeb-synthesis", 1, "fake"),
        ]
        
        for video_dir, label, class_name in video_dirs:
            if not video_dir.exists():
                print(f"⚠ Directory not found: {video_dir}")
                continue
            
            # Find all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
            video_files = []
            for ext in video_extensions:
                video_files.extend(video_dir.glob(f"*{ext}"))
            
            print(f"  Processing {len(video_files)} videos from {video_dir.name}...")
            
            for video_path in sorted(video_files):
                self._extract_video_frames(video_path, label, class_name)
    
    def _extract_video_frames(
        self,
        video_path: Path,
        label: int,
        class_name: str,
    ):
        """Extract frames from a single video."""
        video_name = video_path.stem
        
        # Create output directory for this video's frames
        output_dir = self.cache_dir / class_name / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠ Could not open video: {video_path.name}")
            return
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps == 0:
            print(f"⚠ Invalid FPS for video: {video_path.name}")
            cap.release()
            return
        
        # Calculate frame sampling interval
        frame_interval = int(video_fps / self.fps)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample at specified FPS
            if frame_count % frame_interval == 0:
                # Check max frames limit
                if self.max_frames_per_video and saved_count >= self.max_frames_per_video:
                    break
                
                # Save frame
                frame_filename = f"frame_{saved_count:04d}.jpg"
                frame_path = output_dir / frame_filename
                
                # Convert BGR to RGB and save
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame_rgb).save(frame_path, quality=95)
                
                # Add to samples
                relative_path = frame_path.relative_to(self.cache_dir)
                self.samples.append((str(frame_path), label, video_name))
                
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
    
    def _save_metadata(self):
        """Save metadata about extracted frames."""
        metadata = {
            'fps': self.fps,
            'max_frames_per_video': self.max_frames_per_video,
            'total_frames': len(self.samples),
            'frames': [
                {
                    'path': str(Path(path).relative_to(self.cache_dir)),
                    'label': label,
                    'video': video,
                }
                for path, label, video in self.samples
            ]
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata to: {self.metadata_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load and preprocess frame."""
        frame_path, label, video_name = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(frame_path).convert("RGB")
            
            # Resize (MUST match internal dataset preprocessing)
            image = image.resize((self.resize_size, self.resize_size), Image.BICUBIC)
            
            # Convert to numpy array
            image = np.array(image, dtype=np.float32) / 255.0
            
            # Normalize (MUST match internal dataset)
            if self.normalize:
                image = (image - self.normalize_mean) / self.normalize_std
            
            # Convert to torch tensor (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)
            
            return {
                "image": image,
                "label": torch.tensor(label, dtype=torch.long),
                "video": video_name,
            }
        
        except Exception as e:
            print(f"⚠ Error loading frame {frame_path}: {e}")
            # Return zero tensor as fallback
            return {
                "image": torch.zeros((3, self.resize_size, self.resize_size), dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long),
                "video": video_name,
            }
    
    def get_video_level_labels(self) -> dict:
        """
        Get video-level ground truth labels.
        
        Returns:
            dict mapping video_name to label (0=real, 1=fake)
        """
        video_labels = {}
        for _, label, video_name in self.samples:
            if video_name not in video_labels:
                video_labels[video_name] = label
        return video_labels


def prepare_celebdf_splits(
    root_dir: str,
    output_dir: str = "data/benchmark",
    fps: float = 1.0,
):
    """
    Prepare Celeb-DF dataset and create metadata.
    
    This is a utility function to extract frames and prepare the dataset
    for benchmark evaluation.
    
    Args:
        root_dir: Root directory containing Celeb-DF videos
        output_dir: Output directory for metadata
        fps: Frames per second for extraction
    """
    print("\n" + "=" * 80)
    print("CELEB-DF V2 BENCHMARK PREPARATION")
    print("=" * 80)
    
    # Create dataset (this triggers frame extraction)
    dataset = CelebDFDataset(
        root_dir=root_dir,
        fps=fps,
        resize_size=256,
        normalize=True,
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save statistics
    stats = {
        "total_frames": len(dataset),
        "real_frames": sum(1 for _, label, _ in dataset.samples if label == 0),
        "fake_frames": sum(1 for _, label, _ in dataset.samples if label == 1),
        "num_videos": len(dataset.get_video_level_labels()),
        "fps": fps,
        "cache_dir": str(dataset.cache_dir),
    }
    
    stats_file = output_path / "celebdf_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Preparation complete!")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Real frames:  {stats['real_frames']}")
    print(f"  Fake frames:  {stats['fake_frames']}")
    print(f"  Videos:       {stats['num_videos']}")
    print(f"  Cache dir:    {dataset.cache_dir}")
    print(f"  Stats saved:  {stats_file}")
    print("\n" + "=" * 80)
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Celeb-DF v2 benchmark dataset")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory containing Celeb-DF v2 videos",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark",
        help="Output directory for metadata",
    )
    
    args = parser.parse_args()
    
    prepare_celebdf_splits(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        fps=args.fps,
    )
