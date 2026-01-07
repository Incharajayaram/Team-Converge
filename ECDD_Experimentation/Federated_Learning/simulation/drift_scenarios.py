"""
Drift scenario injection for federated learning simulation.

Implements various attack patterns to simulate distribution drift:
- Sudden attacks (new deepfake methods)
- Gradual shifts (quality degradation)
- Localized attacks (some clients only)
- Correlated attacks (multiple clients simultaneously)

Uses transformations inspired by ECDD edge cases.
"""

import numpy as np
from PIL import Image, ImageFilter
import torch
from typing import List, Dict, Optional, Callable, Tuple
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== Attack Transformations ====================

def apply_blur_attack(image: Image.Image, radius: float = 4.0) -> Image.Image:
    """
    Apply Gaussian blur (simulates blur-based deepfakes).
    
    Args:
        image: PIL Image
        radius: Blur radius (2, 4, 8)
        
    Returns:
        Blurred image
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_jpeg_compression(image: Image.Image, quality: int = 30) -> Image.Image:
    """
    Apply heavy JPEG compression (simulates compression artifacts).
    
    Args:
        image: PIL Image
        quality: JPEG quality (lower = more compression)
        
    Returns:
        Compressed image
    """
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def apply_resize_artifacts(image: Image.Image, downscale_factor: float = 0.5) -> Image.Image:
    """
    Apply resize artifacts (downscale then upscale).
    
    Args:
        image: PIL Image
        downscale_factor: Scale factor (0.5 = half size)
        
    Returns:
        Image with resize artifacts
    """
    original_size = image.size
    small_size = (int(original_size[0] * downscale_factor), 
                  int(original_size[1] * downscale_factor))
    
    # Downscale then upscale
    small_image = image.resize(small_size, Image.BILINEAR)
    return small_image.resize(original_size, Image.BILINEAR)


def apply_multi_transformation(image: Image.Image, 
                               transformations: List[str],
                               intensities: Optional[Dict[str, float]] = None) -> Image.Image:
    """
    Apply multiple transformations in sequence.
    
    Args:
        image: PIL Image
        transformations: List of transformation names
        intensities: Optional intensity parameters
        
    Returns:
        Transformed image
    """
    if intensities is None:
        intensities = {}
    
    result = image.copy()
    
    for transform in transformations:
        if transform == 'blur':
            radius = intensities.get('blur', 4.0)
            result = apply_blur_attack(result, radius=radius)
        elif transform == 'jpeg':
            quality = int(intensities.get('jpeg', 30))
            result = apply_jpeg_compression(result, quality=quality)
        elif transform == 'resize':
            factor = intensities.get('resize', 0.5)
            result = apply_resize_artifacts(result, downscale_factor=factor)
    
    return result


# ==================== Drift Scenario Classes ====================

class DriftScenario:
    """
    Base class for drift scenarios.
    """
    
    def __init__(self, 
                 attack_type: str,
                 intensity: float,
                 start_round: int,
                 affected_clients: List[int]):
        """
        Initialize drift scenario.
        
        Args:
            attack_type: Type of attack ('blur', 'jpeg', 'resize', 'multi')
            intensity: Attack intensity (0-1 or specific parameter)
            start_round: Round when drift starts
            affected_clients: List of client IDs to affect
        """
        self.attack_type = attack_type
        self.intensity = intensity
        self.start_round = start_round
        self.affected_clients = affected_clients
        self.active = False
    
    def should_apply(self, round_num: int, client_id: int) -> bool:
        """
        Check if attack should be applied for given round and client.
        
        Args:
            round_num: Current round number
            client_id: Client ID
            
        Returns:
            True if attack should be applied
        """
        return (round_num >= self.start_round and 
                client_id in self.affected_clients)
    
    def apply(self, image: Image.Image, round_num: int, client_id: int) -> Image.Image:
        """
        Apply attack to image if conditions are met.
        
        Args:
            image: Input image
            round_num: Current round
            client_id: Client ID
            
        Returns:
            Possibly transformed image
        """
        if not self.should_apply(round_num, client_id):
            return image
        
        return self._apply_attack(image, round_num, client_id)
    
    def _apply_attack(self, image: Image.Image, round_num: int, client_id: int) -> Image.Image:
        """Override in subclasses."""
        raise NotImplementedError


class SuddenAttackScenario(DriftScenario):
    """
    Sudden attack: New attack type appears abruptly at specific round.
    
    Example: At round 50, blur-based deepfakes appear for 30% of clients.
    """
    
    def __init__(self,
                 attack_type: str = 'blur',
                 intensity: float = 0.3,
                 start_round: int = 50,
                 affected_clients: Optional[List[int]] = None,
                 affected_ratio: float = 0.3,
                 total_clients: int = 10):
        """
        Initialize sudden attack scenario.
        
        Args:
            attack_type: Attack type
            intensity: Proportion of samples to affect (0-1)
            start_round: Round when attack starts
            affected_clients: Specific clients (or None to sample randomly)
            affected_ratio: Ratio of clients to affect if not specified
            total_clients: Total clients for random sampling
        """
        if affected_clients is None:
            num_affected = int(total_clients * affected_ratio)
            affected_clients = list(np.random.choice(total_clients, num_affected, replace=False))
        
        super().__init__(attack_type, intensity, start_round, affected_clients)
        self.sample_probability = intensity  # Probability of transforming each sample
    
    def _apply_attack(self, image: Image.Image, round_num: int, client_id: int) -> Image.Image:
        """Apply attack with probability = intensity."""
        if np.random.random() < self.sample_probability:
            if self.attack_type == 'blur':
                return apply_blur_attack(image, radius=4.0)
            elif self.attack_type == 'jpeg':
                return apply_jpeg_compression(image, quality=30)
            elif self.attack_type == 'resize':
                return apply_resize_artifacts(image, downscale_factor=0.5)
        
        return image


class GradualDriftScenario(DriftScenario):
    """
    Gradual drift: Attack intensity increases slowly over time.
    
    Example: JPEG quality degrades from 100 to 30 over 50 rounds.
    """
    
    def __init__(self,
                 attack_type: str = 'jpeg',
                 intensity_start: float = 0.0,
                 intensity_end: float = 0.5,
                 start_round: int = 0,
                 duration: int = 50,
                 affected_clients: Optional[List[int]] = None,
                 total_clients: int = 10):
        """
        Initialize gradual drift scenario.
        
        Args:
            attack_type: Attack type
            intensity_start: Starting intensity
            intensity_end: Ending intensity
            start_round: Round when drift starts
            duration: Number of rounds for drift to complete
            affected_clients: Clients to affect (None = all)
            total_clients: Total clients
        """
        if affected_clients is None:
            affected_clients = list(range(total_clients))
        
        super().__init__(attack_type, intensity_end, start_round, affected_clients)
        self.intensity_start = intensity_start
        self.intensity_end = intensity_end
        self.duration = duration
    
    def get_current_intensity(self, round_num: int) -> float:
        """Compute current intensity based on round number."""
        if round_num < self.start_round:
            return self.intensity_start
        
        rounds_since_start = round_num - self.start_round
        if rounds_since_start >= self.duration:
            return self.intensity_end
        
        # Linear interpolation
        progress = rounds_since_start / self.duration
        return self.intensity_start + (self.intensity_end - self.intensity_start) * progress
    
    def _apply_attack(self, image: Image.Image, round_num: int, client_id: int) -> Image.Image:
        """Apply attack with current intensity."""
        current_intensity = self.get_current_intensity(round_num)
        
        # Apply with probability = current_intensity
        if np.random.random() < current_intensity:
            if self.attack_type == 'blur':
                # Increase blur radius over time
                max_radius = 8.0
                radius = 2.0 + (max_radius - 2.0) * current_intensity
                return apply_blur_attack(image, radius=radius)
            
            elif self.attack_type == 'jpeg':
                # Decrease JPEG quality over time
                quality = int(100 - 70 * current_intensity)  # 100 → 30
                return apply_jpeg_compression(image, quality=quality)
            
            elif self.attack_type == 'resize':
                # Increase downscale factor
                factor = 1.0 - 0.5 * current_intensity  # 1.0 → 0.5
                return apply_resize_artifacts(image, downscale_factor=factor)
        
        return image


class LocalizedDriftScenario(DriftScenario):
    """
    Localized drift: Only specific clients (geographic region/platform) affected.
    
    Example: Only clients 0-2 see new attack type.
    """
    
    def __init__(self,
                 attack_type: str = 'blur',
                 intensity: float = 0.5,
                 start_round: int = 30,
                 affected_clients: List[int] = [0, 1, 2]):
        """
        Initialize localized drift scenario.
        
        Args:
            attack_type: Attack type
            intensity: Attack intensity
            start_round: Start round
            affected_clients: Specific clients to affect
        """
        super().__init__(attack_type, intensity, start_round, affected_clients)
    
    def _apply_attack(self, image: Image.Image, round_num: int, client_id: int) -> Image.Image:
        """Apply attack to affected clients only."""
        if np.random.random() < self.intensity:
            if self.attack_type == 'blur':
                return apply_blur_attack(image, radius=4.0)
            elif self.attack_type == 'jpeg':
                return apply_jpeg_compression(image, quality=30)
            elif self.attack_type == 'resize':
                return apply_resize_artifacts(image, downscale_factor=0.5)
        
        return image


class CorrelatedDriftScenario(DriftScenario):
    """
    Correlated drift: Multiple clients see same attack simultaneously.
    
    Example: All clients see blur attack starting at round 40.
    """
    
    def __init__(self,
                 attack_type: str = 'blur',
                 intensity: float = 0.4,
                 start_round: int = 40,
                 affected_clients: Optional[List[int]] = None,
                 correlation: float = 0.8,
                 total_clients: int = 10):
        """
        Initialize correlated drift scenario.
        
        Args:
            attack_type: Attack type
            intensity: Attack intensity
            start_round: Start round
            affected_clients: Clients to affect (None = random sample)
            correlation: Correlation strength (0-1)
            total_clients: Total clients
        """
        if affected_clients is None:
            num_affected = int(total_clients * correlation)
            affected_clients = list(np.random.choice(total_clients, num_affected, replace=False))
        
        super().__init__(attack_type, intensity, start_round, affected_clients)
    
    def _apply_attack(self, image: Image.Image, round_num: int, client_id: int) -> Image.Image:
        """Apply same attack to all affected clients."""
        if np.random.random() < self.intensity:
            if self.attack_type == 'blur':
                return apply_blur_attack(image, radius=6.0)
            elif self.attack_type == 'jpeg':
                return apply_jpeg_compression(image, quality=25)
            elif self.attack_type == 'resize':
                return apply_resize_artifacts(image, downscale_factor=0.4)
        
        return image


# ==================== Utility Functions ====================

def create_scenario_from_config(config: Dict) -> DriftScenario:
    """
    Create drift scenario from configuration dictionary.
    
    Args:
        config: Configuration with scenario parameters
        
    Returns:
        DriftScenario instance
        
    Example:
        >>> config = {
        ...     'type': 'sudden',
        ...     'attack_type': 'blur',
        ...     'intensity': 0.3,
        ...     'start_round': 50,
        ...     'affected_clients': [0, 1, 2]
        ... }
        >>> scenario = create_scenario_from_config(config)
    """
    scenario_type = config['type']
    
    if scenario_type == 'sudden':
        return SuddenAttackScenario(
            attack_type=config.get('attack_type', 'blur'),
            intensity=config.get('intensity', 0.3),
            start_round=config.get('start_round', 50),
            affected_clients=config.get('affected_clients'),
            affected_ratio=config.get('affected_ratio', 0.3),
            total_clients=config.get('total_clients', 10)
        )
    
    elif scenario_type == 'gradual':
        return GradualDriftScenario(
            attack_type=config.get('attack_type', 'jpeg'),
            intensity_start=config.get('intensity_start', 0.0),
            intensity_end=config.get('intensity_end', 0.5),
            start_round=config.get('start_round', 0),
            duration=config.get('duration', 50),
            affected_clients=config.get('affected_clients'),
            total_clients=config.get('total_clients', 10)
        )
    
    elif scenario_type == 'localized':
        return LocalizedDriftScenario(
            attack_type=config.get('attack_type', 'blur'),
            intensity=config.get('intensity', 0.5),
            start_round=config.get('start_round', 30),
            affected_clients=config['affected_clients']
        )
    
    elif scenario_type == 'correlated':
        return CorrelatedDriftScenario(
            attack_type=config.get('attack_type', 'blur'),
            intensity=config.get('intensity', 0.4),
            start_round=config.get('start_round', 40),
            affected_clients=config.get('affected_clients'),
            correlation=config.get('correlation', 0.8),
            total_clients=config.get('total_clients', 10)
        )
    
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


# ==================== Testing ====================

if __name__ == "__main__":
    print("Testing Drift Scenarios...")
    
    # Create test image
    print("\n1. Creating test image:")
    test_image = Image.new('RGB', (224, 224), color='blue')
    print(f"  Test image size: {test_image.size}")
    
    # Test individual attacks
    print("\n2. Testing individual attacks:")
    
    blur_img = apply_blur_attack(test_image, radius=4.0)
    print(f"  Blur: {blur_img.size}")
    
    jpeg_img = apply_jpeg_compression(test_image, quality=30)
    print(f"  JPEG: {jpeg_img.size}")
    
    resize_img = apply_resize_artifacts(test_image, downscale_factor=0.5)
    print(f"  Resize: {resize_img.size}")
    
    # Test scenarios
    print("\n3. Testing sudden attack scenario:")
    scenario = SuddenAttackScenario(
        attack_type='blur',
        intensity=0.3,
        start_round=50,
        affected_clients=[0, 1, 2],
        total_clients=10
    )
    
    # Before start
    img_before = scenario.apply(test_image, round_num=40, client_id=1)
    print(f"  Round 40 (before start): transformed={img_before is not test_image}")
    
    # After start
    img_after = scenario.apply(test_image, round_num=60, client_id=1)
    print(f"  Round 60 (after start): transformed={img_after is not test_image}")
    
    # Unaffected client
    img_unaffected = scenario.apply(test_image, round_num=60, client_id=5)
    print(f"  Round 60 (unaffected client): transformed={img_unaffected is not test_image}")
    
    # Test gradual drift
    print("\n4. Testing gradual drift scenario:")
    scenario = GradualDriftScenario(
        attack_type='jpeg',
        intensity_start=0.0,
        intensity_end=0.5,
        start_round=0,
        duration=100,
        total_clients=10
    )
    
    for round_num in [0, 25, 50, 75, 100]:
        intensity = scenario.get_current_intensity(round_num)
        print(f"  Round {round_num}: intensity={intensity:.3f}")
    
    # Test scenario creation from config
    print("\n5. Testing scenario creation from config:")
    config = {
        'type': 'sudden',
        'attack_type': 'blur',
        'intensity': 0.3,
        'start_round': 50,
        'affected_clients': [0, 1, 2]
    }
    scenario = create_scenario_from_config(config)
    print(f"  Created: {scenario.__class__.__name__}")
    print(f"  Attack type: {scenario.attack_type}")
    print(f"  Affected clients: {scenario.affected_clients}")
    
    print("\n✅ Drift scenarios tests passed!")
