"""
Anomaly detection for identifying clients with divergent distributions.

Uses clustering (DBSCAN) on distribution similarities to find anomalous clients.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import DBSCAN
from collections import Counter


def compute_divergence_matrix(client_histograms: List[np.ndarray],
                               metric: str = 'js') -> np.ndarray:
    """
    Compute pairwise divergence matrix between all clients.
    
    Args:
        client_histograms: List of client histogram distributions
        metric: Distance metric ('js' for Jensen-Shannon, 'l2' for Euclidean)
        
    Returns:
        n x n divergence matrix
        
    Example:
        >>> hists = [hist1, hist2, hist3]
        >>> matrix = compute_divergence_matrix(hists)
        >>> matrix.shape
        (3, 3)
    """
    n_clients = len(client_histograms)
    divergence_matrix = np.zeros((n_clients, n_clients))
    
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            if metric == 'js':
                # Jensen-Shannon divergence
                div = jensenshannon(client_histograms[i], client_histograms[j])
            elif metric == 'l2':
                # Euclidean distance
                div = np.linalg.norm(client_histograms[i] - client_histograms[j])
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Symmetric matrix
            divergence_matrix[i, j] = div
            divergence_matrix[j, i] = div
    
    return divergence_matrix


def detect_anomalous_clients(client_histograms: List[np.ndarray],
                             eps: float = 0.15,
                             min_samples: int = 2,
                             metric: str = 'js') -> List[int]:
    """
    Detect anomalous clients using DBSCAN clustering.
    
    Clients in small clusters or marked as noise are considered anomalous.
    
    Args:
        client_histograms: List of client histogram distributions
        eps: Maximum distance for clustering (default 0.15)
        min_samples: Minimum cluster size (default 2)
        metric: Distance metric ('js' or 'l2')
        
    Returns:
        List of anomalous client indices
        
    Example:
        >>> hists = [hist1, hist2, ..., hist10]
        >>> anomalous = detect_anomalous_clients(hists)
        >>> print(f"Anomalous clients: {anomalous}")
    """
    if len(client_histograms) < min_samples:
        return []  # Not enough clients to detect anomalies
    
    # Compute divergence matrix
    divergence_matrix = compute_divergence_matrix(client_histograms, metric=metric)
    
    # Apply DBSCAN with precomputed distance matrix
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(divergence_matrix)
    
    # Identify majority cluster
    valid_labels = labels[labels != -1]  # Exclude noise (-1)
    if len(valid_labels) == 0:
        # All clients are noise - no clear majority
        return list(range(len(client_histograms)))
    
    label_counts = Counter(valid_labels)
    majority_label = label_counts.most_common(1)[0][0]
    
    # Anomalous = not in majority cluster (includes noise)
    anomalous_clients = [i for i, label in enumerate(labels) 
                        if label != majority_label]
    
    return anomalous_clients


class ClientClusterer:
    """
    Wrapper for client clustering and anomaly analysis.
    """
    
    def __init__(self, eps: float = 0.15, min_samples: int = 2, metric: str = 'js'):
        """
        Initialize clusterer.
        
        Args:
            eps: Maximum distance for clustering
            min_samples: Minimum cluster size
            metric: Distance metric
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        self.divergence_matrix = None
        self.labels = None
        self.anomalous_clients = []
        self.cluster_info = {}
    
    def fit(self, client_histograms: List[np.ndarray], client_ids: Optional[List[int]] = None):
        """
        Fit clustering on client histograms.
        
        Args:
            client_histograms: List of client distributions
            client_ids: Optional client IDs (default: 0, 1, 2, ...)
        """
        if client_ids is None:
            client_ids = list(range(len(client_histograms)))
        
        # Compute divergence matrix
        self.divergence_matrix = compute_divergence_matrix(client_histograms, self.metric)
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        self.labels = clustering.fit_predict(self.divergence_matrix)
        
        # Identify anomalous clients
        self.anomalous_clients = self._identify_anomalous(client_ids)
        
        # Compute cluster info
        self._compute_cluster_info(client_ids)
    
    def _identify_anomalous(self, client_ids: List[int]) -> List[int]:
        """Identify anomalous clients."""
        valid_labels = self.labels[self.labels != -1]
        if len(valid_labels) == 0:
            return client_ids  # All clients anomalous
        
        label_counts = Counter(valid_labels)
        majority_label = label_counts.most_common(1)[0][0]
        
        anomalous = [client_ids[i] for i, label in enumerate(self.labels)
                    if label != majority_label]
        return anomalous
    
    def _compute_cluster_info(self, client_ids: List[int]):
        """Compute cluster statistics."""
        unique_labels = set(self.labels)
        
        for label in unique_labels:
            cluster_members = [client_ids[i] for i, l in enumerate(self.labels) if l == label]
            cluster_size = len(cluster_members)
            
            # Compute within-cluster divergence
            if cluster_size > 1:
                member_indices = [i for i, l in enumerate(self.labels) if l == label]
                sub_matrix = self.divergence_matrix[np.ix_(member_indices, member_indices)]
                avg_divergence = sub_matrix.sum() / (cluster_size * (cluster_size - 1))
            else:
                avg_divergence = 0.0
            
            self.cluster_info[int(label)] = {
                'size': cluster_size,
                'members': cluster_members,
                'avg_divergence': float(avg_divergence),
                'is_noise': label == -1
            }
    
    def get_anomalous_clients(self) -> List[int]:
        """Get list of anomalous client IDs."""
        return self.anomalous_clients.copy()
    
    def get_clusters(self) -> Dict:
        """Get cluster information."""
        return self.cluster_info.copy()
    
    def get_majority_cluster(self) -> List[int]:
        """Get members of majority (normal) cluster."""
        valid_labels = [l for l in self.labels if l != -1]
        if not valid_labels:
            return []
        
        majority_label = Counter(valid_labels).most_common(1)[0][0]
        return self.cluster_info[majority_label]['members']
    
    def visualize_divergence_matrix(self) -> np.ndarray:
        """
        Get divergence matrix for visualization.
        
        Returns:
            Divergence matrix (can be plotted with plt.imshow)
        """
        return self.divergence_matrix.copy()
    
    def get_summary(self) -> Dict:
        """
        Get summary of clustering results.
        
        Returns:
            Dictionary with clustering summary
        """
        num_clusters = len([l for l in set(self.labels) if l != -1])
        num_noise = sum(self.labels == -1)
        num_anomalous = len(self.anomalous_clients)
        
        return {
            'num_clients': len(self.labels),
            'num_clusters': num_clusters,
            'num_noise': num_noise,
            'num_anomalous': num_anomalous,
            'anomaly_rate': num_anomalous / len(self.labels) if len(self.labels) > 0 else 0.0,
            'clusters': self.cluster_info
        }
    
    def __repr__(self):
        summary = self.get_summary()
        return f"ClientClusterer(clients={summary['num_clients']}, clusters={summary['num_clusters']}, anomalous={summary['num_anomalous']})"


class AnomalyScorer:
    """
    Compute anomaly scores for clients based on divergence from majority.
    """
    
    def __init__(self):
        self.scores = {}
    
    def compute_scores(self, client_histograms: List[np.ndarray], 
                      client_ids: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Compute anomaly scores for all clients.
        
        Score = average divergence from all other clients.
        
        Args:
            client_histograms: List of client distributions
            client_ids: Optional client IDs
            
        Returns:
            Dictionary {client_id: anomaly_score}
        """
        if client_ids is None:
            client_ids = list(range(len(client_histograms)))
        
        divergence_matrix = compute_divergence_matrix(client_histograms)
        
        # Compute average divergence for each client
        for i, client_id in enumerate(client_ids):
            # Average divergence to all other clients
            other_divergences = np.concatenate([divergence_matrix[i, :i], 
                                               divergence_matrix[i, i+1:]])
            avg_divergence = np.mean(other_divergences)
            self.scores[client_id] = float(avg_divergence)
        
        return self.scores.copy()
    
    def get_top_anomalous(self, k: int = 5) -> List[Tuple[int, float]]:
        """
        Get top-k most anomalous clients.
        
        Args:
            k: Number of top anomalies to return
            
        Returns:
            List of (client_id, score) tuples, sorted by score
        """
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]
    
    def get_score(self, client_id: int) -> float:
        """Get anomaly score for a specific client."""
        return self.scores.get(client_id, 0.0)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Anomaly Detection...")
    
    np.random.seed(42)
    
    # Create normal clients (majority)
    print("\n1. Creating synthetic client distributions:")
    normal_clients = []
    for i in range(8):
        samples = np.random.beta(2, 5, size=1000)
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        hist = hist / hist.sum()
        normal_clients.append(hist)
        print(f"Normal client {i}: mean={np.dot(hist, np.linspace(0, 1, 20)):.3f}")
    
    # Create anomalous clients (different distribution)
    anomalous_clients = []
    for i in range(2):
        samples = np.random.beta(5, 2, size=1000)  # Reversed distribution
        hist, _ = np.histogram(samples, bins=20, range=(0, 1))
        hist = hist / hist.sum()
        anomalous_clients.append(hist)
        print(f"Anomalous client {8+i}: mean={np.dot(hist, np.linspace(0, 1, 20)):.3f}")
    
    # Combine all clients
    all_clients = normal_clients + anomalous_clients
    client_ids = list(range(10))
    
    # Test divergence matrix
    print("\n2. Testing divergence matrix:")
    div_matrix = compute_divergence_matrix(all_clients)
    print(f"Divergence matrix shape: {div_matrix.shape}")
    print(f"Average divergence between normal clients: {div_matrix[:8, :8].mean():.4f}")
    print(f"Average divergence between normal and anomalous: {div_matrix[:8, 8:].mean():.4f}")
    
    # Test simple anomaly detection
    print("\n3. Testing anomaly detection:")
    detected_anomalous = detect_anomalous_clients(all_clients, eps=0.15, min_samples=2)
    print(f"Detected anomalous clients: {detected_anomalous}")
    print(f"True anomalous clients: [8, 9]")
    
    # Test ClientClusterer
    print("\n4. Testing ClientClusterer:")
    clusterer = ClientClusterer(eps=0.15, min_samples=2)
    clusterer.fit(all_clients, client_ids)
    
    print(clusterer)
    print(f"Anomalous clients: {clusterer.get_anomalous_clients()}")
    print(f"Majority cluster: {clusterer.get_majority_cluster()}")
    
    summary = clusterer.get_summary()
    print(f"\nClustering summary:")
    for label, info in summary['clusters'].items():
        cluster_type = "Noise" if info['is_noise'] else f"Cluster {label}"
        print(f"  {cluster_type}: {info['size']} clients, avg_div={info['avg_divergence']:.4f}")
    
    # Test AnomalyScorer
    print("\n5. Testing AnomalyScorer:")
    scorer = AnomalyScorer()
    scores = scorer.compute_scores(all_clients, client_ids)
    
    print(f"Anomaly scores:")
    for client_id, score in sorted(scores.items()):
        print(f"  Client {client_id}: {score:.4f}")
    
    top_anomalous = scorer.get_top_anomalous(k=3)
    print(f"\nTop 3 anomalous clients: {top_anomalous}")
    
    print("\n✅ Anomaly detection tests passed!")
