"""
ECDD Dataset Loader
PyTorch/TensorFlow compatible data loading for training.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
import random

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using NumPy-only mode")

from augmentations import AugmentationPipeline, preprocess_for_training


class ECDDDataset:
    """
    ECDD Dataset loader compatible with both PyTorch and pure NumPy.
    
    Follows locked preprocessing from Phase 1:
    - PIL decoder
    - LANCZOS resize to 256x256
    - ImageNet normalization
    """
    
    def __init__(self, 
                 data_dir: Path,
                 split: str = "train",
                 augment: bool = True,
                 target_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            data_dir: Root directory containing processed data
            split: One of "train", "finetune", "calibration", "test_seen", "test_unseen"
            augment: Whether to apply augmentations
            target_size: Target image size (default 256x256)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.target_size = target_size
        
        # Load split manifest
        self.images = []
        self.labels = []
        self._load_split()
        
        # Augmentation pipeline
        self.aug_pipeline = AugmentationPipeline() if augment else None
    
    def _load_split(self):
        """Load images and labels for the specified split."""
        split_dir = self.data_dir / "splits" / self.split
        
        if not split_dir.exists():
            # Fallback: load from real/fake directories
            real_dir = self.data_dir / "real"
            fake_dir = self.data_dir / "fake"
            
            if real_dir.exists():
                for img in real_dir.glob("*"):
                    if img.suffix.lower() in ['.jpg', '.png', '.webp']:
                        self.images.append(img)
                        self.labels.append(0)  # 0 = real
            
            if fake_dir.exists():
                for img in fake_dir.glob("*"):
                    if img.suffix.lower() in ['.jpg', '.png', '.webp']:
                        self.images.append(img)
                        self.labels.append(1)  # 1 = fake
        else:
            # Load from split directory
            for img in split_dir.glob("*"):
                if img.suffix.lower() in ['.jpg', '.png', '.webp']:
                    self.images.append(img)
                    # Infer label from filename or parent
                    label = 1 if "fake" in str(img).lower() else 0
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images for split '{self.split}'")
        print(f"  Real: {self.labels.count(0)}, Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get preprocessed image and label."""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation if enabled
        if self.augment and self.aug_pipeline:
            image, _ = self.aug_pipeline.apply_random(image)
        
        # Preprocess (matches Phase 1 locked pipeline)
        tensor = preprocess_for_training(image, self.target_size)
        
        return tensor, label
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of preprocessed images and labels (NumPy mode)."""
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)
        
        batch_indices = indices[:batch_size]
        
        images = []
        labels = []
        
        for idx in batch_indices:
            img, label = self[idx]
            images.append(img)
            labels.append(label)
        
        return np.stack(images), np.array(labels)
    
    def as_generator(self, batch_size: int, shuffle: bool = True):
        """Yield batches as a generator."""
        indices = list(range(len(self)))
        
        while True:
            if shuffle:
                random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                images = []
                labels = []
                
                for idx in batch_indices:
                    img, label = self[idx]
                    images.append(img)
                    labels.append(label)
                
                yield np.stack(images), np.array(labels)


if TORCH_AVAILABLE:
    class ECDDTorchDataset(Dataset):
        """PyTorch Dataset wrapper for ECDD data."""
        
        def __init__(self, ecdd_dataset: ECDDDataset):
            self.dataset = ecdd_dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            
            # Convert to torch tensors
            # Image: HWC -> CHW
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            label = torch.tensor(label, dtype=torch.float32)
            
            return image, label
    
    
    def create_dataloader(data_dir: Path,
                          split: str,
                          batch_size: int = 32,
                          shuffle: bool = True,
                          num_workers: int = 0,
                          augment: bool = True) -> DataLoader:
        """Create a PyTorch DataLoader for ECDD data."""
        
        base_dataset = ECDDDataset(data_dir, split=split, augment=augment)
        torch_dataset = ECDDTorchDataset(base_dataset)
        
        return DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


def create_splits(processed_dir: Path, train_ratio: float = 0.7):
    """Create train/test splits from processed data."""
    processed_dir = Path(processed_dir)
    
    real_images = list((processed_dir / "real").glob("*.[jp][pn][g]"))
    fake_images = list((processed_dir / "fake").glob("*.[jp][pn][g]"))
    
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    # Calculate split sizes
    n_real_train = int(len(real_images) * train_ratio)
    n_fake_train = int(len(fake_images) * train_ratio)
    
    splits = {
        "train": real_images[:n_real_train] + fake_images[:n_fake_train],
        "test_seen": real_images[n_real_train:] + fake_images[n_fake_train:]
    }
    
    # Create split directories and symlinks
    for split_name, images in splits.items():
        split_dir = processed_dir / "splits" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for img in images:
            # Create symlink or copy
            dst = split_dir / img.name
            if not dst.exists():
                dst.symlink_to(img)
    
    print(f"Created splits: train={len(splits['train'])}, test={len(splits['test_seen'])}")
    return splits


if __name__ == "__main__":
    print("ECDD Dataset Loader")
    print("="*40)
    
    # Test with existing data
    data_dir = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation\ECDD_Experiment_Data")
    
    if data_dir.exists():
        dataset = ECDDDataset(data_dir, split="train", augment=False)
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample, label = dataset[0]
            print(f"Sample shape: {sample.shape}")
            print(f"Sample label: {label}")
            print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
    else:
        print(f"Data directory not found: {data_dir}")
