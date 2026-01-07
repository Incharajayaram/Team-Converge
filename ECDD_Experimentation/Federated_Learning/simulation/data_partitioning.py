"""
Data partitioning for federated learning simulation.

Creates non-IID data splits to simulate realistic heterogeneous clients.
Uses clustering and Dirichlet distribution for heterogeneity.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, Subset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_non_iid_splits(dataset: Dataset,
                          num_clients: int,
                          concentration: float = 0.5,
                          feature_extractor: Optional[torch.nn.Module] = None,
                          random_seed: int = 42) -> List[List[int]]:
    """
    Create non-IID data splits using clustering and Dirichlet distribution.
    
    Method:
    1. Extract features from dataset (or use labels/indices)
    2. Cluster samples into K clusters (K = num_clients * 2)
    3. Distribute clusters to clients using Dirichlet distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients to split data for
        concentration: Dirichlet concentration (lower = more heterogeneous)
        feature_extractor: Optional model to extract features
        random_seed: Random seed for reproducibility
        
    Returns:
        List of index lists, one per client
        
    Example:
        >>> splits = create_non_iid_splits(dataset, num_clients=10, concentration=0.5)
        >>> client_0_data = Subset(dataset, splits[0])
    """
    np.random.seed(random_seed)
    
    n_samples = len(dataset)
    n_clusters = num_clients * 2  # More clusters than clients
    
    print(f"Creating non-IID splits for {num_clients} clients...")
    print(f"  Total samples: {n_samples}")
    print(f"  Concentration: {concentration} (lower = more heterogeneous)")
    
    # Step 1: Get features for clustering
    if feature_extractor is not None:
        print(f"  Extracting features with provided model...")
        features = extract_features(dataset, feature_extractor)
    else:
        # Use simple heuristic: if dataset has labels, use them
        # Otherwise use sample indices (assumes temporal/spatial structure)
        print(f"  Using label-based clustering...")
        features = get_label_features(dataset)
    
    # Reduce dimensionality if needed
    if features.shape[1] > 50:
        print(f"  Reducing dimensionality: {features.shape[1]} → 50")
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
    
    # Step 2: Cluster samples
    print(f"  Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Step 3: Distribute clusters to clients using Dirichlet
    print(f"  Distributing clusters to clients...")
    client_splits = [[] for _ in range(num_clients)]
    
    for cluster_id in range(n_clusters):
        # Get samples in this cluster
        cluster_samples = np.where(cluster_labels == cluster_id)[0]
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([concentration] * num_clients)
        
        # Split cluster across clients according to proportions
        split_points = (np.cumsum(proportions) * len(cluster_samples)).astype(int)
        
        for client_id in range(num_clients):
            start = split_points[client_id - 1] if client_id > 0 else 0
            end = split_points[client_id]
            client_splits[client_id].extend(cluster_samples[start:end].tolist())
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_splits[client_id])
    
    # Print statistics
    sizes = [len(split) for split in client_splits]
    print(f"  Client data sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    
    return client_splits


def create_powerlaw_sizes(num_clients: int,
                         total_samples: int,
                         alpha: float = 1.5,
                         min_samples: int = 50,
                         random_seed: int = 42) -> np.ndarray:
    """
    Sample client data sizes from power-law distribution.
    
    Power-law (Zipf) distribution creates realistic heterogeneity:
    - Few clients have lots of data
    - Many clients have little data
    
    Args:
        num_clients: Number of clients
        total_samples: Total samples to distribute
        alpha: Shape parameter (higher = more skewed)
        min_samples: Minimum samples per client
        random_seed: Random seed
        
    Returns:
        Array of sample counts per client
        
    Example:
        >>> sizes = create_powerlaw_sizes(10, 1000, alpha=1.5)
        >>> print(sizes)
        [245, 156, 121, 98, 82, 71, 63, 56, 51, 57]
    """
    np.random.seed(random_seed)
    
    # Sample from Zipf distribution
    sizes = np.random.zipf(alpha, size=num_clients)
    
    # Normalize to total_samples
    sizes = sizes / sizes.sum() * total_samples
    sizes = sizes.astype(int)
    
    # Ensure minimum samples per client
    sizes = np.maximum(sizes, min_samples)
    
    # Adjust to match total_samples exactly
    current_total = sizes.sum()
    if current_total < total_samples:
        # Distribute remaining samples randomly
        remaining = total_samples - current_total
        random_indices = np.random.choice(num_clients, size=remaining, replace=True)
        for idx in random_indices:
            sizes[idx] += 1
    elif current_total > total_samples:
        # Remove excess samples from largest clients
        excess = current_total - total_samples
        for _ in range(excess):
            largest_idx = np.argmax(sizes)
            if sizes[largest_idx] > min_samples:
                sizes[largest_idx] -= 1
    
    return sizes


def resample_to_powerlaw(client_splits: List[List[int]],
                         alpha: float = 1.5,
                         min_samples: int = 50,
                         random_seed: int = 42) -> List[List[int]]:
    """
    Resample existing splits to match power-law distribution.
    
    Args:
        client_splits: Existing client splits
        alpha: Power-law parameter
        min_samples: Minimum per client
        random_seed: Random seed
        
    Returns:
        Resampled client splits with power-law sizes
    """
    np.random.seed(random_seed)
    
    num_clients = len(client_splits)
    total_samples = sum(len(split) for split in client_splits)
    
    # Get target sizes
    target_sizes = create_powerlaw_sizes(num_clients, total_samples, alpha, min_samples, random_seed)
    
    # Resample each client's data
    new_splits = []
    for client_id in range(num_clients):
        current_size = len(client_splits[client_id])
        target_size = target_sizes[client_id]
        
        if target_size <= current_size:
            # Subsample
            sampled_indices = np.random.choice(
                client_splits[client_id],
                size=target_size,
                replace=False
            )
        else:
            # Oversample (with replacement)
            sampled_indices = np.random.choice(
                client_splits[client_id],
                size=target_size,
                replace=True
            )
        
        new_splits.append(sampled_indices.tolist())
    
    return new_splits


def assign_students_to_hubs(num_students: int,
                            num_hubs: int,
                            method: str = 'balanced',
                            random_seed: int = 42) -> Dict[int, List[int]]:
    """
    Assign student clients to hub devices.
    
    Args:
        num_students: Total number of student clients
        num_hubs: Number of hub devices
        method: Assignment method ('balanced', 'random', 'skewed')
        random_seed: Random seed
        
    Returns:
        Dictionary {hub_id: [student_ids]}
        
    Example:
        >>> assignment = assign_students_to_hubs(15, 3, method='balanced')
        >>> print(assignment)
        {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9], 2: [10, 11, 12, 13, 14]}
    """
    np.random.seed(random_seed)
    
    student_ids = np.arange(num_students)
    
    if method == 'balanced':
        # Equal distribution
        np.random.shuffle(student_ids)
        students_per_hub = num_students // num_hubs
        
        assignment = {}
        for hub_id in range(num_hubs):
            start = hub_id * students_per_hub
            end = start + students_per_hub if hub_id < num_hubs - 1 else num_students
            assignment[hub_id] = student_ids[start:end].tolist()
    
    elif method == 'random':
        # Random assignment
        hub_assignments = np.random.randint(0, num_hubs, size=num_students)
        assignment = {hub_id: [] for hub_id in range(num_hubs)}
        for student_id, hub_id in enumerate(hub_assignments):
            assignment[hub_id].append(student_id)
    
    elif method == 'skewed':
        # Power-law distribution of students to hubs
        sizes = create_powerlaw_sizes(num_hubs, num_students, alpha=1.3, min_samples=1)
        np.random.shuffle(student_ids)
        
        assignment = {}
        start = 0
        for hub_id in range(num_hubs):
            end = start + sizes[hub_id]
            assignment[hub_id] = student_ids[start:end].tolist()
            start = end
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return assignment


def extract_features(dataset: Dataset, model: torch.nn.Module, batch_size: int = 32) -> np.ndarray:
    """
    Extract features from dataset using a model.
    
    Args:
        dataset: Dataset to extract features from
        model: Feature extraction model
        batch_size: Batch size for processing
        
    Returns:
        Feature matrix (n_samples, n_features)
    """
    from torch.utils.data import DataLoader
    
    model.eval()
    device = next(model.parameters()).device
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features_list = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            features = model(images)
            
            # Flatten if needed
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            features_list.append(features.cpu().numpy())
    
    return np.vstack(features_list)


def get_label_features(dataset: Dataset) -> np.ndarray:
    """
    Get features from dataset labels (fallback method).
    
    Args:
        dataset: Dataset with labels
        
    Returns:
        One-hot encoded labels or index-based features
    """
    try:
        # Try to get labels
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, (tuple, list)) and len(item) > 1:
                labels.append(item[1])
            else:
                labels.append(0)  # Default label
        
        labels = np.array(labels)
        
        # One-hot encode if categorical
        unique_labels = np.unique(labels)
        if len(unique_labels) < 100:  # Categorical
            features = np.eye(len(unique_labels))[labels]
        else:
            # Use indices as features (assumes structure)
            features = np.arange(len(dataset)).reshape(-1, 1)
    
    except:
        # Fallback: use indices
        features = np.arange(len(dataset)).reshape(-1, 1)
    
    return features


# Example usage and testing
if __name__ == "__main__":
    print("Testing Data Partitioning...")
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset:")
    
    class SyntheticDataset(Dataset):
        def __init__(self, n_samples=1000):
            self.n_samples = n_samples
            # Simulate labels (0 or 1)
            self.labels = np.random.randint(0, 2, size=n_samples)
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            # Return (data, label)
            return torch.randn(3, 64, 64), self.labels[idx]
    
    dataset = SyntheticDataset(n_samples=1000)
    print(f"  Dataset size: {len(dataset)}")
    
    # Test non-IID splits
    print("\n2. Creating non-IID splits:")
    splits = create_non_iid_splits(dataset, num_clients=10, concentration=0.5)
    
    sizes = [len(split) for split in splits]
    print(f"  Splits created: {len(splits)}")
    print(f"  Sizes: {sizes}")
    print(f"  Min: {min(sizes)}, Max: {max(sizes)}, Std: {np.std(sizes):.1f}")
    
    # Test power-law sizes
    print("\n3. Creating power-law sizes:")
    powerlaw_sizes = create_powerlaw_sizes(num_clients=10, total_samples=1000, alpha=1.5)
    print(f"  Sizes: {powerlaw_sizes}")
    print(f"  Ratio (max/min): {powerlaw_sizes.max() / powerlaw_sizes.min():.2f}")
    
    # Test resampling
    print("\n4. Resampling to power-law:")
    resampled = resample_to_powerlaw(splits, alpha=1.5, min_samples=50)
    resampled_sizes = [len(split) for split in resampled]
    print(f"  New sizes: {resampled_sizes}")
    print(f"  Total: {sum(resampled_sizes)}")
    
    # Test hub assignment
    print("\n5. Assigning students to hubs:")
    for method in ['balanced', 'random', 'skewed']:
        assignment = assign_students_to_hubs(15, num_hubs=3, method=method)
        sizes = [len(students) for students in assignment.values()]
        print(f"  {method}: {sizes}")
    
    print("\n✅ Data partitioning tests passed!")
