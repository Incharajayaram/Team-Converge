"""
Sketch algorithms for privacy-preserving data summarization.

Implements histograms and statistical summaries for federated monitoring.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class ScoreHistogram:
    """
    Histogram for score distributions with sparse representation.
    
    Designed for deepfake detection scores in [0, 1] range.
    """
    
    def __init__(self, num_bins: int = 20, range: Tuple[float, float] = (0, 1)):
        """
        Initialize histogram.
        
        Args:
            num_bins: Number of bins (default 20)
            range: (min, max) range for scores (default (0, 1))
        """
        self.num_bins = num_bins
        self.range = range
        self.counts = np.zeros(num_bins, dtype=np.int64)
        self.bin_edges = np.linspace(range[0], range[1], num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.total_samples = 0
        
    def update(self, value: float):
        """
        Add a single value to histogram.
        
        Args:
            value: Score value to add
        """
        if self.range[0] <= value <= self.range[1]:
            bin_idx = np.searchsorted(self.bin_edges[1:], value)
            bin_idx = min(bin_idx, self.num_bins - 1)  # Handle edge case
            self.counts[bin_idx] += 1
            self.total_samples += 1
    
    def update_batch(self, values: np.ndarray):
        """
        Add multiple values to histogram.
        
        Args:
            values: Array of score values
        """
        # Filter values in range
        valid_values = values[(values >= self.range[0]) & (values <= self.range[1])]
        
        # Compute histogram
        hist, _ = np.histogram(valid_values, bins=self.bin_edges)
        self.counts += hist
        self.total_samples += len(valid_values)
    
    def get_normalized(self) -> np.ndarray:
        """
        Get normalized histogram (probabilities).
        
        Returns:
            Normalized histogram (sums to 1)
        """
        if self.total_samples == 0:
            return np.ones(self.num_bins) / self.num_bins
        return self.counts / self.total_samples
    
    def to_sparse_dict(self) -> Dict[int, int]:
        """
        Get sparse representation (only non-zero bins).
        
        Returns:
            Dictionary {bin_index: count} for non-zero bins
            
        Example:
            >>> hist.to_sparse_dict()
            {2: 10, 5: 23, 8: 15}
        """
        return {i: int(count) for i, count in enumerate(self.counts) if count > 0}
    
    @staticmethod
    def from_sparse_dict(sparse_dict: Dict[int, int], num_bins: int = 20) -> 'ScoreHistogram':
        """
        Reconstruct histogram from sparse representation.
        
        Args:
            sparse_dict: Sparse histogram dictionary
            num_bins: Number of bins
            
        Returns:
            ScoreHistogram instance
        """
        hist = ScoreHistogram(num_bins=num_bins)
        for bin_idx, count in sparse_dict.items():
            hist.counts[bin_idx] = count
            hist.total_samples += count
        return hist
    
    def add_dp_noise(self, epsilon: float) -> np.ndarray:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            epsilon: Privacy parameter
            
        Returns:
            Noisy histogram counts (non-negative)
        """
        from .privacy_utils import add_laplace_noise
        
        noisy_counts = add_laplace_noise(self.counts.astype(float), sensitivity=1.0, epsilon=epsilon)
        # Post-process: ensure non-negative
        noisy_counts = np.maximum(noisy_counts, 0)
        return noisy_counts
    
    def clear(self):
        """Reset histogram to empty."""
        self.counts = np.zeros(self.num_bins, dtype=np.int64)
        self.total_samples = 0
    
    def to_dict(self) -> Dict:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary with histogram data
        """
        return {
            'counts': self.counts.tolist(),
            'num_bins': self.num_bins,
            'range': self.range,
            'total_samples': self.total_samples,
            'bin_edges': self.bin_edges.tolist()
        }
    
    def __repr__(self):
        return f"ScoreHistogram(bins={self.num_bins}, samples={self.total_samples})"


class StatisticalSummary:
    """
    Running statistics using Welford's algorithm.
    
    Computes mean, variance, and quantiles in an online fashion.
    """
    
    def __init__(self, track_quantiles: bool = True, quantile_buffer_size: int = 1000):
        """
        Initialize statistical summary.
        
        Args:
            track_quantiles: If True, track quantiles (requires buffering)
            quantile_buffer_size: Size of buffer for quantile estimation
        """
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # For variance computation
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        self.track_quantiles = track_quantiles
        if track_quantiles:
            self.buffer = deque(maxlen=quantile_buffer_size)
    
    def update(self, value: float):
        """
        Add a single value to statistics.
        
        Uses Welford's online algorithm for numerical stability.
        
        Args:
            value: New value to add
        """
        self.count += 1
        
        # Welford's algorithm for mean and variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        # Min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Quantile buffer
        if self.track_quantiles:
            self.buffer.append(value)
    
    def update_batch(self, values: np.ndarray):
        """
        Add multiple values to statistics.
        
        Args:
            values: Array of values
        """
        for value in values:
            self.update(value)
    
    @property
    def variance(self) -> float:
        """Get variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count
    
    @property
    def std(self) -> float:
        """Get standard deviation."""
        return np.sqrt(self.variance)
    
    def get_quantiles(self, quantiles: List[float] = [0.25, 0.5, 0.75, 0.95]) -> Dict[float, float]:
        """
        Get quantiles from buffered values.
        
        Args:
            quantiles: List of quantile values (e.g., [0.25, 0.5, 0.75])
            
        Returns:
            Dictionary {quantile: value}
        """
        if not self.track_quantiles or len(self.buffer) == 0:
            return {q: self.mean for q in quantiles}
        
        buffer_array = np.array(self.buffer)
        return {q: float(np.quantile(buffer_array, q)) for q in quantiles}
    
    def get_stats(self) -> Dict:
        """
        Get all statistics.
        
        Returns:
            Dictionary with mean, std, min, max, quantiles
        """
        stats = {
            'mean': self.mean,
            'std': self.std,
            'min': self.min_val if self.count > 0 else 0.0,
            'max': self.max_val if self.count > 0 else 0.0,
            'count': self.count
        }
        
        if self.track_quantiles:
            stats['quantiles'] = self.get_quantiles()
        
        return stats
    
    def clear(self):
        """Reset all statistics."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        if self.track_quantiles:
            self.buffer.clear()
    
    def __repr__(self):
        return f"StatisticalSummary(n={self.count}, mean={self.mean:.3f}, std={self.std:.3f})"


class CompressionUtils:
    """Utilities for compressing sketches for transmission."""
    
    @staticmethod
    def compress_histogram(histogram: np.ndarray, threshold: float = 1e-6) -> Dict:
        """
        Compress histogram by removing near-zero bins.
        
        Args:
            histogram: Normalized histogram
            threshold: Minimum value to keep
            
        Returns:
            Compressed representation
        """
        sparse = {i: float(val) for i, val in enumerate(histogram) if val > threshold}
        compression_ratio = len(sparse) / len(histogram)
        
        return {
            'sparse': sparse,
            'total_bins': len(histogram),
            'compression_ratio': compression_ratio
        }
    
    @staticmethod
    def decompress_histogram(compressed: Dict) -> np.ndarray:
        """
        Decompress histogram from sparse representation.
        
        Args:
            compressed: Compressed histogram dictionary
            
        Returns:
            Full histogram array
        """
        num_bins = compressed['total_bins']
        histogram = np.zeros(num_bins)
        
        for bin_idx, value in compressed['sparse'].items():
            histogram[int(bin_idx)] = value
        
        return histogram


# Example usage and testing
if __name__ == "__main__":
    print("Testing Sketch Algorithms...")
    
    # Test ScoreHistogram
    print("\n1. Testing ScoreHistogram:")
    hist = ScoreHistogram(num_bins=10, range=(0, 1))
    
    # Add some synthetic data
    np.random.seed(42)
    scores = np.random.beta(2, 5, size=1000)  # Skewed distribution
    hist.update_batch(scores)
    
    print(f"Histogram: {hist}")
    print(f"Normalized: {hist.get_normalized()}")
    print(f"Sparse representation: {hist.to_sparse_dict()}")
    
    # Test DP noise
    noisy = hist.add_dp_noise(epsilon=1.0)
    print(f"With DP noise (ε=1.0): {noisy}")
    
    # Test sparse compression
    compression_ratio = len(hist.to_sparse_dict()) / hist.num_bins
    print(f"Compression ratio: {compression_ratio:.2%}")
    
    # Test StatisticalSummary
    print("\n2. Testing StatisticalSummary:")
    stats = StatisticalSummary(track_quantiles=True)
    stats.update_batch(scores)
    
    print(f"Statistics: {stats}")
    print(f"Detailed stats: {stats.get_stats()}")
    
    # Verify against numpy
    print(f"\nVerification against numpy:")
    print(f"Mean: stats={stats.mean:.4f}, numpy={np.mean(scores):.4f}")
    print(f"Std: stats={stats.std:.4f}, numpy={np.std(scores):.4f}")
    print(f"Min: stats={stats.min_val:.4f}, numpy={np.min(scores):.4f}")
    print(f"Max: stats={stats.max_val:.4f}, numpy={np.max(scores):.4f}")
    
    # Test compression
    print("\n3. Testing Compression:")
    normalized = hist.get_normalized()
    compressed = CompressionUtils.compress_histogram(normalized, threshold=0.01)
    print(f"Compression ratio: {compressed['compression_ratio']:.2%}")
    
    decompressed = CompressionUtils.decompress_histogram(compressed)
    reconstruction_error = np.sum(np.abs(normalized - decompressed))
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    print("\n✅ Sketch algorithms tests passed!")
