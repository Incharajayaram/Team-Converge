"""Base dataset class with resize-only, deterministic loading."""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


class BaseDataset(Dataset):
    """
    Base dataset class for deepfake detection.
    Handles deterministic image loading with resize-only preprocessing.

    Supports two loading modes:
    1. Directory-based: Loads from root_dir/real/ and root_dir/fake/
    2. CSV-based: Loads from CSV split files with path and label columns
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_format: str = "jpg",
        resize_size: int = 256,  # CHANGED from 224 to 256
        normalize: bool = True,
        normalize_mean=None,
        normalize_std=None,
        split_file: str = None,  # Optional CSV split file
    ):
        """
        Args:
            root_dir: Root directory containing 'real' and 'fake' subdirectories
            split: 'train', 'val', or 'test'
            image_format: Image file extension (default: 'jpg')
            resize_size: Target image size for resizing (deterministic resize with bicubic)
            normalize: Whether to normalize images with ImageNet constants
            normalize_mean: Normalization mean (ImageNet default if None)
            normalize_std: Normalization std (ImageNet default if None)
            split_file: Optional path to CSV split file (takes precedence over directory-based)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_format = image_format
        self.resize_size = resize_size
        self.normalize = normalize
        self.split_file = split_file

        if normalize_mean is None:
            self.normalize_mean = np.array([0.485, 0.456, 0.406])
        else:
            self.normalize_mean = np.array(normalize_mean)

        if normalize_std is None:
            self.normalize_std = np.array([0.229, 0.224, 0.225])
        else:
            self.normalize_std = np.array(normalize_std)

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Load image paths and labels from CSV or directory structure."""
        if self.split_file:
            self._load_from_csv()
        else:
            self._load_from_directory()

    def _load_from_csv(self):
        """Load samples from CSV split file."""
        split_path = Path(self.split_file)

        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        try:
            df = pd.read_csv(split_path)
            for _, row in df.iterrows():
                img_path = row['path']
                label = int(row['label'])
                # Verify file exists before adding
                if Path(img_path).exists():
                    self.samples.append((img_path, label))
            print(f"✓ Loaded {len(self.samples)} samples from {split_path}")
        except Exception as e:
            print(f"⚠ Error loading split file: {e}")
            raise

    def _load_from_directory(self):
        """Load image paths and labels from directory structure."""
        from PIL import UnidentifiedImageError

        corrupted_count = 0

        # Load real images (label 0)
        real_dir = self.root_dir / "real"
        if real_dir.exists():
            for img_path in sorted(real_dir.glob(f"*.{self.image_format}")):
                # Verify image is valid before adding
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.samples.append((str(img_path), 0))
                except (UnidentifiedImageError, IOError, OSError):
                    corrupted_count += 1
                    print(f"⚠ Skipping corrupted image: {img_path.name}")

        # Load fake images (label 1)
        fake_dir = self.root_dir / "fake"
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob(f"*.{self.image_format}")):
                # Verify image is valid before adding
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.samples.append((str(img_path), 1))
                except (UnidentifiedImageError, IOError, OSError):
                    corrupted_count += 1
                    print(f"⚠ Skipping corrupted image: {img_path.name}")

        if corrupted_count > 0:
            print(f"⚠ Skipped {corrupted_count} corrupted image(s) during loading")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load image and return (image, label)."""
        from PIL import UnidentifiedImageError

        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")

            # Resize
            image = image.resize((self.resize_size, self.resize_size), Image.BICUBIC)

            # Convert to numpy array
            image = np.array(image, dtype=np.float32) / 255.0

            # Normalize
            if self.normalize:
                image = (image - self.normalize_mean) / self.normalize_std

            # Convert to torch tensor
            image = torch.from_numpy(image).permute(2, 0, 1)

            return {"image": image, "label": torch.tensor(label, dtype=torch.long)}

        except (UnidentifiedImageError, IOError, OSError) as e:
            # Fallback: Return a black image if file is corrupted
            print(f"⚠ Error loading {img_path}: {type(e).__name__}")
            # Return zero tensor as fallback
            return {
                "image": torch.zeros((3, self.resize_size, self.resize_size), dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long)
            }
