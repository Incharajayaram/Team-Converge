"""
ECDD Data Augmentation Pipeline
Implements deployment-simulation augmentations for robust training.
"""

import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import io
import random

@dataclass
class AugmentConfig:
    """Configuration for augmentation pipeline."""
    jpeg_qualities: List[int] = None
    resize_scales: List[float] = None
    blur_sigmas: List[float] = None
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    def __post_init__(self):
        if self.jpeg_qualities is None:
            self.jpeg_qualities = [30, 50, 75, 95]
        if self.resize_scales is None:
            self.resize_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        if self.blur_sigmas is None:
            self.blur_sigmas = [0.0, 0.5, 1.0, 2.0]


# Default config matching implementation plan
DEFAULT_CONFIG = AugmentConfig()


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression at specified quality."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def resize_chain(image: Image.Image, down_scale: float, up_scale: float = None) -> Image.Image:
    """Simulate screenshot/re-upload by downscaling then upscaling."""
    if up_scale is None:
        up_scale = 1.0 / down_scale
    
    original_size = image.size
    
    # Downscale
    new_size = (int(original_size[0] * down_scale), int(original_size[1] * down_scale))
    image = image.resize(new_size, Image.Resampling.BILINEAR)
    
    # Upscale back to original
    image = image.resize(original_size, Image.Resampling.BILINEAR)
    
    return image


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur."""
    if sigma <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def brightness_contrast(image: Image.Image, 
                        brightness: float = 1.0, 
                        contrast: float = 1.0) -> Image.Image:
    """Adjust brightness and contrast."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    arr = np.array(image, dtype=np.float32)
    
    # Brightness
    arr = arr * brightness
    
    # Contrast (around mean)
    mean = arr.mean()
    arr = (arr - mean) * contrast + mean
    
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class AugmentationPipeline:
    """Pipeline for applying deployment-simulation augmentations."""
    
    def __init__(self, config: AugmentConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    def random_jpeg(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Apply random JPEG compression."""
        quality = random.choice(self.config.jpeg_qualities)
        return jpeg_compress(image, quality), {"jpeg_quality": quality}
    
    def random_resize_chain(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Apply random resize chain (screenshot simulation)."""
        scale = random.choice([s for s in self.config.resize_scales if s != 1.0])
        return resize_chain(image, scale), {"resize_scale": scale}
    
    def random_blur(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Apply random Gaussian blur."""
        sigma = random.choice(self.config.blur_sigmas)
        return gaussian_blur(image, sigma), {"blur_sigma": sigma}
    
    def random_color(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Apply random brightness/contrast adjustment."""
        b = random.uniform(*self.config.brightness_range)
        c = random.uniform(*self.config.contrast_range)
        return brightness_contrast(image, b, c), {"brightness": b, "contrast": c}
    
    def apply_random(self, image: Image.Image, 
                     p_jpeg: float = 0.5,
                     p_resize: float = 0.3,
                     p_blur: float = 0.3,
                     p_color: float = 0.3) -> Tuple[Image.Image, dict]:
        """Apply random combination of augmentations."""
        augmentations_applied = {}
        
        if random.random() < p_jpeg:
            image, info = self.random_jpeg(image)
            augmentations_applied.update(info)
        
        if random.random() < p_resize:
            image, info = self.random_resize_chain(image)
            augmentations_applied.update(info)
        
        if random.random() < p_blur:
            image, info = self.random_blur(image)
            augmentations_applied.update(info)
        
        if random.random() < p_color:
            image, info = self.random_color(image)
            augmentations_applied.update(info)
        
        return image, augmentations_applied
    
    def apply_all_variants(self, image: Image.Image) -> List[Tuple[Image.Image, str]]:
        """Generate all augmentation variants of an image."""
        variants = [
            (image.copy(), "original")
        ]
        
        # JPEG variants
        for q in self.config.jpeg_qualities:
            variants.append((jpeg_compress(image, q), f"jpeg_q{q}"))
        
        # Resize chain variants
        for s in self.config.resize_scales:
            if s != 1.0:
                variants.append((resize_chain(image, s), f"resize_{s}x"))
        
        # Blur variants
        for sigma in self.config.blur_sigmas:
            if sigma > 0:
                variants.append((gaussian_blur(image, sigma), f"blur_s{sigma}"))
        
        return variants


def generate_augmented_dataset(input_dir: Path, 
                               output_dir: Path,
                               variants_per_image: int = 5) -> int:
    """Generate augmented dataset from source images."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = AugmentationPipeline()
    generated = 0
    
    for img_path in input_dir.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.webp']:
            continue
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            for i in range(variants_per_image):
                aug_img, info = pipeline.apply_random(image)
                
                # Generate output filename
                stem = img_path.stem
                suffix = "_".join(f"{k}{v}" for k, v in info.items()) or "orig"
                out_name = f"{stem}_aug{i}_{suffix[:20]}.jpg"
                
                aug_img.save(output_dir / out_name, quality=95)
                generated += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return generated


# Preprocessing functions matching Phase 1 locked parameters
def preprocess_for_training(image: Image.Image, 
                            target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Preprocess image for model training (matches inference pipeline)."""
    
    # Resize using LANCZOS (locked in Phase 1)
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # To numpy array
    arr = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    arr = arr / 255.0
    
    # Apply ImageNet normalization (locked in Phase 1)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    
    return arr


if __name__ == "__main__":
    print("ECDD Augmentation Pipeline")
    print("="*40)
    
    # Example usage
    config = AugmentConfig(
        jpeg_qualities=[30, 50, 75, 95],
        resize_scales=[0.5, 0.75, 1.25, 1.5],
        blur_sigmas=[0.5, 1.0, 2.0]
    )
    
    print(f"JPEG qualities: {config.jpeg_qualities}")
    print(f"Resize scales: {config.resize_scales}")
    print(f"Blur sigmas: {config.blur_sigmas}")
    print("\nReady for training!")
