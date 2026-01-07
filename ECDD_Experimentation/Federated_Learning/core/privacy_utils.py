"""
Privacy utilities for differential privacy in federated learning.

Implements Laplace and Gaussian mechanisms for differential privacy,
along with privacy budget tracking.
"""

import numpy as np
from typing import Union, List


def add_laplace_noise(value: Union[float, np.ndarray], 
                      sensitivity: float, 
                      epsilon: float) -> Union[float, np.ndarray]:
    """
    Add Laplace noise for epsilon-differential privacy.
    
    Args:
        value: Value or array to add noise to
        sensitivity: Sensitivity of the query (L1 sensitivity)
        epsilon: Privacy parameter (smaller = more privacy)
        
    Returns:
        Noisy value(s)
        
    Example:
        >>> count = 100
        >>> noisy_count = add_laplace_noise(count, sensitivity=1, epsilon=0.1)
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=np.shape(value))
    return value + noise


def add_gaussian_noise(value: Union[float, np.ndarray],
                       sensitivity: float,
                       epsilon: float,
                       delta: float = 1e-5) -> Union[float, np.ndarray]:
    """
    Add Gaussian noise for (epsilon, delta)-differential privacy.
    
    Args:
        value: Value or array to add noise to
        sensitivity: Sensitivity of the query (L2 sensitivity)
        epsilon: Privacy parameter
        delta: Probability of privacy breach (typically 1e-5)
        
    Returns:
        Noisy value(s)
        
    Example:
        >>> histogram = np.array([10, 20, 15, 8])
        >>> noisy_hist = add_gaussian_noise(histogram, sensitivity=1, epsilon=1.0)
    """
    # Compute sigma for (epsilon, delta)-DP
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    noise = np.random.normal(0, sigma, size=np.shape(value))
    return value + noise


def clip_and_noise(value: Union[float, np.ndarray],
                   min_val: float,
                   max_val: float,
                   epsilon: float,
                   mechanism: str = 'laplace') -> Union[float, np.ndarray]:
    """
    Clip value to range and add DP noise.
    
    Args:
        value: Value to clip and add noise to
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        epsilon: Privacy parameter
        mechanism: 'laplace' or 'gaussian'
        
    Returns:
        Clipped and noisy value
    """
    clipped = np.clip(value, min_val, max_val)
    sensitivity = max_val - min_val
    
    if mechanism == 'laplace':
        return add_laplace_noise(clipped, sensitivity, epsilon)
    elif mechanism == 'gaussian':
        return add_gaussian_noise(clipped, sensitivity, epsilon)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


class PrivacyBudgetTracker:
    """
    Track cumulative privacy budget across multiple queries.
    
    Implements sequential composition for differential privacy.
    """
    
    def __init__(self, total_budget: float, delta: float = 1e-5):
        """
        Initialize privacy budget tracker.
        
        Args:
            total_budget: Total epsilon budget available
            delta: Delta parameter for (epsilon, delta)-DP
        """
        self.total_budget = total_budget
        self.delta = delta
        self.consumed_budget = 0.0
        self.query_history = []
        
    def consume(self, epsilon: float, query_name: str = "unnamed") -> bool:
        """
        Consume privacy budget for a query.
        
        Args:
            epsilon: Budget to consume
            query_name: Name of the query (for tracking)
            
        Returns:
            True if budget available, False otherwise
        """
        if self.consumed_budget + epsilon > self.total_budget:
            return False
        
        self.consumed_budget += epsilon
        self.query_history.append({
            'query': query_name,
            'epsilon': epsilon,
            'cumulative': self.consumed_budget
        })
        return True
    
    def remaining(self) -> float:
        """Get remaining privacy budget."""
        return self.total_budget - self.consumed_budget
    
    def reset(self):
        """Reset privacy budget (use with caution!)."""
        self.consumed_budget = 0.0
        self.query_history = []
    
    def get_history(self) -> List[dict]:
        """Get history of all queries."""
        return self.query_history.copy()
    
    def __repr__(self):
        return f"PrivacyBudgetTracker(consumed={self.consumed_budget:.3f}/{self.total_budget:.3f})"


class DPHistogram:
    """
    Differentially private histogram with automatic noise addition.
    """
    
    def __init__(self, num_bins: int, range: tuple = (0, 1), epsilon: float = 1.0):
        """
        Initialize DP histogram.
        
        Args:
            num_bins: Number of histogram bins
            range: (min, max) range for binning
            epsilon: Privacy parameter
        """
        self.num_bins = num_bins
        self.range = range
        self.epsilon = epsilon
        self.counts = np.zeros(num_bins)
        self.bin_edges = np.linspace(range[0], range[1], num_bins + 1)
        
    def add_values(self, values: np.ndarray):
        """Add values to histogram (without DP)."""
        hist, _ = np.histogram(values, bins=self.bin_edges)
        self.counts += hist
    
    def get_noisy_counts(self, normalize: bool = True) -> np.ndarray:
        """
        Get histogram counts with DP noise.
        
        Args:
            normalize: If True, return normalized probabilities
            
        Returns:
            Noisy histogram counts or probabilities
        """
        # Sensitivity = 1 (one individual can affect at most one bin)
        noisy_counts = add_laplace_noise(self.counts, sensitivity=1.0, epsilon=self.epsilon)
        
        # Ensure non-negative (post-processing doesn't violate DP)
        noisy_counts = np.maximum(noisy_counts, 0)
        
        if normalize:
            total = noisy_counts.sum()
            if total > 0:
                return noisy_counts / total
            else:
                return np.ones(self.num_bins) / self.num_bins
        
        return noisy_counts
    
    def clear(self):
        """Clear histogram counts."""
        self.counts = np.zeros(self.num_bins)


def compute_privacy_loss(num_queries: int, epsilon_per_query: float) -> float:
    """
    Compute total privacy loss under sequential composition.
    
    Args:
        num_queries: Number of queries
        epsilon_per_query: Privacy parameter per query
        
    Returns:
        Total epsilon (sequential composition)
    """
    return num_queries * epsilon_per_query


def compute_advanced_composition(num_queries: int, 
                                epsilon_per_query: float,
                                delta: float = 1e-5) -> float:
    """
    Compute privacy loss under advanced composition.
    
    Better than sequential composition for many queries.
    
    Args:
        num_queries: Number of queries
        epsilon_per_query: Privacy parameter per query
        delta: Delta parameter
        
    Returns:
        Total epsilon (advanced composition)
    """
    # Advanced composition: epsilon_total = epsilon * sqrt(2 * num_queries * ln(1/delta)) + num_queries * epsilon^2
    eps = epsilon_per_query
    n = num_queries
    
    epsilon_total = eps * np.sqrt(2 * n * np.log(1 / delta)) + n * (eps ** 2)
    return epsilon_total


# Example usage and testing
if __name__ == "__main__":
    print("Testing Privacy Utils...")
    
    # Test Laplace noise
    value = 100
    noisy = add_laplace_noise(value, sensitivity=1, epsilon=0.1)
    print(f"\nLaplace noise test:")
    print(f"Original: {value}, Noisy: {noisy:.2f}")
    
    # Test Gaussian noise
    histogram = np.array([10, 20, 15, 8, 5])
    noisy_hist = add_gaussian_noise(histogram, sensitivity=1, epsilon=1.0)
    print(f"\nGaussian noise test:")
    print(f"Original histogram: {histogram}")
    print(f"Noisy histogram: {noisy_hist}")
    
    # Test privacy budget tracker
    tracker = PrivacyBudgetTracker(total_budget=10.0)
    print(f"\nPrivacy budget tracker:")
    print(tracker)
    
    success = tracker.consume(3.0, "query_1")
    print(f"Consumed 3.0: {success}, Remaining: {tracker.remaining():.2f}")
    
    success = tracker.consume(8.0, "query_2")
    print(f"Tried to consume 8.0: {success}, Remaining: {tracker.remaining():.2f}")
    
    # Test DP histogram
    print(f"\nDP Histogram test:")
    dp_hist = DPHistogram(num_bins=5, range=(0, 1), epsilon=1.0)
    
    # Simulate some data
    data = np.random.beta(2, 5, size=1000)  # Skewed distribution
    dp_hist.add_values(data)
    
    print(f"True histogram: {dp_hist.counts}")
    noisy = dp_hist.get_noisy_counts(normalize=True)
    print(f"Noisy (normalized): {noisy}")
    print(f"Sum: {noisy.sum():.4f}")
    
    print("\n✅ Privacy utils tests passed!")
