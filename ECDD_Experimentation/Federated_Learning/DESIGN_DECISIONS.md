# Federated Drift Detection - Design Decisions

This document captures all key design decisions for the federated privacy-preserving drift detection system.

---

## 1. Privacy vs Utility Trade-off

### Decision
- **Sketch type**: Histograms only (20 bins for score distribution)
- **Statistics**: Mean, std, quantiles (p25, p50, p75, p95)
- **Differential Privacy**: ε = 1.0 (moderate privacy)
- **Communication strategy**: Clients only send when they have sufficient local data (≥100 samples)

### Rationale
- Histograms are simple, interpretable, and sufficient for drift detection
- Starting with ε=1.0 balances privacy and utility; can be tuned in experiments
- More sophisticated sketches (Count-Min, HyperLogLog) add complexity without clear benefit initially
- Local accumulation reduces communication overhead and improves privacy (larger batch = less leakage)

### Implementation Notes
- Use uniform binning: [0.0-0.05, 0.05-0.1, ..., 0.95-1.0] for 20 bins
- Apply Laplace noise to histogram counts: count + Lap(1/ε)
- Track window size and only transmit when window_size >= 100
- Statistics computed on raw scores, then DP noise added

### Future Extensions
- Experiment with different ε values (0.1, 1.0, 10.0, ∞) in privacy-utility trade-off study
- Add Count-Min Sketch for heavy-hitter detection if needed
- Adaptive binning based on score distribution

---

## 2. Drift Detection Algorithms

### Decision
Use **ensemble of multiple detectors**:
1. **KS-test (Kolmogorov-Smirnov)**: For sudden distribution shifts
2. **PSI (Population Stability Index)**: For gradual drift
3. **JS-divergence (Jensen-Shannon)**: For distribution similarity comparison

### Rationale
- Different algorithms catch different drift types
- KS-test: Non-parametric, good for sudden changes, p-value threshold = 0.01
- PSI: Industry standard for monitoring, threshold = 0.1 (small drift), 0.25 (significant drift)
- JS-divergence: Symmetric, bounded [0,1], threshold = 0.1
- Ensemble reduces false positives (require 2/3 detectors to agree)

### Implementation Notes
```python
# KS-test
from scipy.stats import ks_2samp
statistic, p_value = ks_2samp(baseline_scores, current_scores)
drift_detected = p_value < 0.01

# PSI
psi = sum((current_prop - baseline_prop) * ln(current_prop / baseline_prop))
drift_detected = psi > 0.1

# JS-divergence
from scipy.spatial.distance import jensenshannon
js_div = jensenshannon(baseline_hist, current_hist)
drift_detected = js_div > 0.1
```

### Avoid
- ML-based drift detectors (autoencoders, VAEs): Too complex, hard to interpret, require training
- Single detector: Prone to false positives/negatives

### Future Extensions
- Sequential change detection (CUSUM, Page-Hinkley) for online monitoring
- ADWIN (ADaptive WINdowing) for automatic window adjustment

---

## 3. Baseline Distribution Management

### Decision
- **Type**: Static global baseline
- **Source**: Computed from training/validation data distribution
- **Representation**: Histogram (20 bins) + statistical moments
- **Shared across**: All clients use same baseline initially

### Rationale
- Static baseline is simple and reproducible
- Training data distribution is known and represents "expected" behavior
- Global baseline enables comparing all clients to same reference
- Avoids complexity of sliding window baselines (concept drift handling)

### Implementation Notes
```python
# Compute baseline from validation set
baseline_scores = []
for batch in val_loader:
    outputs = model(batch)
    scores = torch.sigmoid(outputs)
    baseline_scores.extend(scores.cpu().numpy())

baseline_hist, bin_edges = np.histogram(baseline_scores, bins=20, range=(0, 1))
baseline_hist = baseline_hist / baseline_hist.sum()  # Normalize

baseline_stats = {
    'mean': np.mean(baseline_scores),
    'std': np.std(baseline_scores),
    'quantiles': np.quantile(baseline_scores, [0.25, 0.5, 0.75, 0.95])
}
```

### Future Extensions (Phase 2)
- **Sliding baseline**: Update baseline with recent data (weighted moving average)
- **Per-client baselines**: Allow personalization for non-IID settings
- **Multi-modal baselines**: Separate baselines for different data types (platforms, demographics)

---


## 4. Anomaly Detection Strategy

### Decision
- **Method**: DBSCAN clustering on distribution embeddings
- **Features**: JS-divergence vectors between each client and all others
- **Anomaly criterion**: Clients forming separate clusters from majority
- **Threshold**: Require at least 2 clients in anomalous cluster (avoid false positives from single outlier)
- **Action**: Alert operator + flag samples for manual review

### Rationale
- Clustering finds groups of clients seeing similar drift (coordinated attack signal)
- DBSCAN handles arbitrary cluster shapes and identifies outliers naturally
- Multiple clients with similar anomaly = stronger signal than single outlier
- JS-divergence captures distribution similarity without raw data
- Manual review loop keeps human in decision process (safety)

### Implementation Notes
\\\python
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import jensenshannon

# Compute pairwise JS-divergence matrix
n_clients = len(client_histograms)
divergence_matrix = np.zeros((n_clients, n_clients))
for i in range(n_clients):
    for j in range(i+1, n_clients):
        div = jensenshannon(client_histograms[i], client_histograms[j])
        divergence_matrix[i, j] = div
        divergence_matrix[j, i] = div

# Cluster clients
clustering = DBSCAN(eps=0.15, min_samples=2, metric='precomputed')
labels = clustering.fit_predict(divergence_matrix)

# Identify anomalous cluster (small cluster separate from majority)
majority_label = mode(labels[labels != -1])
anomalous_clients = [i for i, label in enumerate(labels) if label != majority_label and label != -1]
\\\

### Future Extensions
- Reputation scoring: Track historical accuracy per client, weight by reputation
- Temporal correlation: Track if anomalies persist over multiple rounds
- Severity scoring: Combine cluster size, divergence magnitude, persistence

---

## 5. Adaptive Thresholding Mechanism

### Decision
- **Scope**: Global threshold (same for all clients)
- **Update frequency**: Only when drift detected (not every round)
- **Optimization metric**: Fixed False Positive Rate (FPR = 0.01)
- **Method**: Compute optimal threshold from aggregated score distributions
- **No veto**: Clients must accept server threshold (simplicity)

### Rationale
- Global threshold ensures consistency across deployment
- Updating only on drift reduces instability and communication overhead
- FPR constraint = production requirement (false positives costly for user trust)
- Server has global view, better positioned to set threshold than individual clients
- Veto mechanism adds complexity and potential for exploitation

### Implementation Notes
\\\python
def calibrate_threshold(aggregated_histogram, target_fpr=0.01):
    # Reconstruct approximate CDF from histogram
    bin_centers = np.linspace(0.025, 0.975, 20)  # Bin centers
    counts = aggregated_histogram
    cdf = np.cumsum(counts) / np.sum(counts)
    
    # Find threshold that achieves target FPR
    # Assuming high scores = fake, threshold separates real (low) from fake (high)
    threshold_idx = np.searchsorted(cdf, 1 - target_fpr)
    threshold = bin_centers[threshold_idx]
    
    return threshold
\\\

### Future Extensions (Phase 2)
- Personalized thresholds: Allow per-client adjustment based on local data distribution
- Multi-objective optimization: Balance FPR, FNR, and abstention rate
- Federated threshold voting: Clients propose thresholds, server aggregates

---

## 6. Communication Protocol

### Decision
- **Architecture**: Push-based (clients initiate)
- **Synchronization**: Asynchronous (server doesn't wait for all clients)
- **Frequency**: Clients send after accumulating N predictions (N=500)
- **Compression**: Sparse histogram representation (only non-zero bins)
- **Transport**: REST API (simple, widely supported)

### Rationale
- Push-based reduces server load and gives clients control over timing
- Asynchronous handles stragglers gracefully (real-world networks are unreliable)
- Event-driven (N predictions) rather than time-based is more consistent
- Sparse representation: Most histograms have empty bins (compression ratio ~2-3x)
- REST over gRPC: Simpler for prototyping, easier edge device integration

### Implementation Notes
\\\python
# Client-side
class FederatedClient:
    def __init__(self, server_url, client_id, window_size=500):
        self.server_url = server_url
        self.client_id = client_id
        self.window_size = window_size
        self.prediction_count = 0
        self.monitor = ClientMonitor()
    
    def on_prediction(self, score, confidence, is_ood, abstained):
        self.monitor.update(score, confidence, is_ood, abstained)
        self.prediction_count += 1
        
        if self.prediction_count >= self.window_size:
            sketch = self.monitor.get_sketch()
            self.send_to_server(sketch)
            self.prediction_count = 0
    
    def send_to_server(self, sketch):
        # Compress: only send non-zero histogram bins
        sparse_hist = {i: count for i, count in enumerate(sketch['histogram']) if count > 0}
        payload = {
            'client_id': self.client_id,
            'histogram': sparse_hist,
            'stats': sketch['stats'],
            'timestamp': time.time()
        }
        requests.post(f'{self.server_url}/submit_sketch', json=payload)

# Server-side
@app.route('/submit_sketch', methods=['POST'])
def submit_sketch():
    data = request.json
    client_id = data['client_id']
    
    # Decompress histogram
    sparse_hist = data['histogram']
    full_hist = np.zeros(20)
    for idx, count in sparse_hist.items():
        full_hist[int(idx)] = count
    
    # Store and process
    drift_server.receive_sketch(client_id, full_hist, data['stats'])
    return {'status': 'received'}
\\\

### Future Extensions
- gRPC for production (better performance, streaming)
- Adaptive frequency: Send more often during drift, less during stable periods
- Delta encoding: Only send changes from previous sketch

---


## 7. Simulation Realism

### Decision
- **Data distribution**: Non-IID (cluster test samples by video source, assign clusters to clients)
- **Client heterogeneity**: Power-law distribution for data sizes (realistic: few clients have lots of data, many have little)
- **Temporal dynamics**: 20% random client dropout per round (simulate unreliable networks)
- **Attack injection**: Use ECDD edge cases (blur, compression, JPEG quality) as "new attacks"

### Rationale
- Real-world data is non-IID (different platforms, demographics, capture conditions)
- Power-law matches real user behavior (Pareto principle: 80/20 rule)
- Client dropout is common in federated settings (mobile devices, network issues)
- ECDD edge cases provide realistic distribution shifts without needing new deepfake datasets

### Implementation Notes
\\\python
# Non-IID data partitioning
def create_non_iid_splits(dataset, num_clients, concentration=0.5):
    """
    Cluster samples by video source, assign clusters to clients
    concentration: lower = more heterogeneous
    """
    from sklearn.cluster import KMeans
    
    # Extract features for clustering (e.g., using pre-trained model)
    features = extract_features(dataset)
    
    # Cluster into num_clients * 2 clusters (more clusters than clients)
    kmeans = KMeans(n_clusters=num_clients * 2)
    cluster_labels = kmeans.fit_predict(features)
    
    # Assign clusters to clients using Dirichlet distribution
    from numpy.random import dirichlet
    client_splits = [[] for _ in range(num_clients)]
    for cluster_id in range(num_clients * 2):
        cluster_samples = np.where(cluster_labels == cluster_id)[0]
        proportions = dirichlet([concentration] * num_clients)
        
        # Split cluster across clients according to proportions
        split_points = (np.cumsum(proportions) * len(cluster_samples)).astype(int)
        for i in range(num_clients):
            start = split_points[i-1] if i > 0 else 0
            end = split_points[i]
            client_splits[i].extend(cluster_samples[start:end])
    
    return client_splits

# Power-law data sizes
def sample_powerlaw_sizes(num_clients, total_samples, alpha=1.5):
    """
    Sample client data sizes from power-law distribution
    alpha: shape parameter (1.5 = realistic heterogeneity)
    """
    from numpy.random import zipf
    
    # Sample from Zipf distribution
    sizes = zipf(alpha, num_clients)
    sizes = sizes / sizes.sum() * total_samples
    sizes = sizes.astype(int)
    
    # Ensure all clients have at least min_size samples
    min_size = 50
    sizes = np.maximum(sizes, min_size)
    
    return sizes

# Client dropout simulation
def simulate_round_with_dropout(clients, dropout_rate=0.2):
    """
    Randomly drop clients each round
    """
    active_clients = np.random.choice(
        clients, 
        size=int(len(clients) * (1 - dropout_rate)),
        replace=False
    )
    return active_clients

# Attack injection using ECDD edge cases
def inject_attack(client_data, attack_type='blur', intensity=0.3):
    """
    Apply ECDD-style transformations to simulate new attack
    attack_type: 'blur', 'jpeg', 'compression', 'resize'
    intensity: proportion of samples to affect
    """
    from PIL import Image, ImageFilter
    
    num_affected = int(len(client_data) * intensity)
    affected_indices = np.random.choice(len(client_data), num_affected, replace=False)
    
    for idx in affected_indices:
        img_path, label = client_data[idx]
        img = Image.open(img_path)
        
        if attack_type == 'blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=4))
        elif attack_type == 'jpeg':
            # Simulate heavy JPEG compression
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=30)
            buffer.seek(0)
            img = Image.open(buffer)
        
        # Save modified image (or use on-the-fly transform)
        client_data[idx] = (img, label)
    
    return client_data
\\\

### Simulation Parameters
- **Number of clients**: 10 (baseline), 20, 50, 100 (scalability experiments)
- **Rounds**: 100 rounds per experiment
- **Attack injection timing**: Round 50 (halfway through)
- **Affected clients**: 30% of clients (3/10 in baseline)

---

## 8. Evaluation Metrics

### Decision
**Detection Metrics**:
- **Detection Latency**: Rounds from injection to server exceeding drift threshold
- **Client Identification**: Precision/Recall on identifying affected client IDs
- **False Alarm Rate**: Drift alerts when no actual drift present

**Privacy Metrics**:
- **Privacy Budget**: Cumulative e consumption over rounds
- **Reconstruction Error**: Attempt to reconstruct raw scores from sketches (measure information leakage)

**Efficiency Metrics**:
- **Communication Overhead**: Bytes transmitted per client per round
- **Computation Time**: Server aggregation time vs number of clients

**Success Criterion**:
- Detection = server drift score exceeds threshold within K rounds (K=10)
- Client identification = precision = 0.8, recall = 0.7

### Rationale
- Detection latency measures responsiveness (critical for emerging attacks)
- Client identification validates anomaly detection accuracy
- False alarm rate = practical usability (too many false alerts ? ignored)
- Privacy metrics validate DP guarantees
- Communication overhead = deployment feasibility (edge networks have limited bandwidth)

### Implementation Notes
\\\python
class ExperimentMetrics:
    def __init__(self, injection_round, affected_clients):
        self.injection_round = injection_round
        self.affected_clients = set(affected_clients)
        self.detection_round = None
        self.false_alarms = 0
        self.identified_clients = set()
    
    def update(self, round_num, drift_detected, flagged_clients):
        # Detection latency
        if drift_detected and self.detection_round is None and round_num >= self.injection_round:
            self.detection_round = round_num
        
        # False alarms (drift detected before injection)
        if drift_detected and round_num < self.injection_round:
            self.false_alarms += 1
        
        # Client identification
        if drift_detected:
            self.identified_clients.update(flagged_clients)
    
    def compute_metrics(self, total_rounds):
        # Detection latency
        if self.detection_round:
            latency = self.detection_round - self.injection_round
        else:
            latency = total_rounds  # Failed to detect
        
        # Client identification precision/recall
        true_positives = len(self.identified_clients & self.affected_clients)
        false_positives = len(self.identified_clients - self.affected_clients)
        false_negatives = len(self.affected_clients - self.identified_clients)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'detection_latency': latency,
            'client_precision': precision,
            'client_recall': recall,
            'client_f1': f1,
            'false_alarms': self.false_alarms
        }
\\\

### Baselines for Comparison
1. **Centralized monitoring**: All data aggregated to server (no privacy)
2. **Isolated monitoring**: Each client detects drift independently (no collaboration)
3. **No monitoring**: Fixed threshold, no adaptation

### Statistical Rigor
- **Runs per experiment**: 10 runs with different random seeds
- **Confidence intervals**: Report mean � std or 95% CI
- **Statistical tests**: Paired t-test for comparing methods

---

## 9. Integration with Existing System

### Decision
- **Approach**: Augment existing monitoring (not replace)
- **Integration point**: Wrap existing inference pipeline with FederatedClient
- **Baseline source**: Use existing 	hreshold_calibration.json as starting point
- **Deployment demo**: Simulation only initially, real demo (Pi + Nicla) as future work

### Rationale
- Existing monitoring works well, federated adds collaboration layer
- Minimal changes to existing codebase (wrapper pattern)
- Existing calibration provides validated baseline
- Simulation faster for paper experiments; real demo for follow-up validation

### Implementation Notes
\\\python
# Integration wrapper
from deployment.pi_server.app import detect_deepfake  # Existing function

class FederatedDeepfakeDetector:
    def __init__(self, model, server_url, client_id):
        self.model = model
        self.fed_client = FederatedClient(server_url, client_id)
        
        # Load existing threshold calibration
        with open('outputs/checkpoints_two_stage/calibration/threshold_calibration.json') as f:
            calibration = json.load(f)
            self.local_threshold = calibration['optimal_threshold']
    
    def detect(self, image):
        # Existing detection logic
        score, confidence, is_ood = detect_deepfake(self.model, image)
        
        # Check against local threshold
        prediction = score > self.local_threshold
        
        # Send to federated monitoring
        abstained = confidence < 0.7  # Example abstention logic
        self.fed_client.on_prediction(score, confidence, is_ood, abstained)
        
        return prediction, score, confidence
    
    def update_threshold(self, new_threshold):
        """Called when server broadcasts new threshold"""
        self.local_threshold = new_threshold
\\\

### Future Real Deployment
- Multiple Raspberry Pi devices acting as federated clients
- Central server aggregating drift signals
- Arduino Nicla devices as lightweight clients (send sketches via Pi gateway)

---

## 10. Research Positioning

### Decision
- **Target venue**: ACM Multimedia 2026 or WACV 2027 (realistic tier-1 venues)
- **Backup venues**: FG 2026, ICME 2027
- **Main angle**: "Privacy-Preserving Collaborative Drift Detection for Deepfake Forensics"
- **Novelty claim**: First system combining federated learning + drift detection + adaptive thresholding specifically for deepfake detection with privacy guarantees

### Rationale
- ACM MM / WACV: Strong multimedia/vision venues, accept systems papers
- CVPR/ICCV too competitive for first paper (acceptance rate ~25%)
- ACM MM acceptance rate ~30%, values practical systems with solid experiments
- Combination novelty: Individual pieces exist, but not integrated for this problem domain

### Paper Structure
1. **Introduction**
   - Motivation: Deepfakes evolving, centralized monitoring inadequate
   - Challenge: Privacy constraints prevent data sharing
   - Solution: Federated drift detection with sketch-based monitoring
   
2. **Related Work**
   - Deepfake detection (LaDeDa, attention mechanisms)
   - Federated learning (FedAvg, FedNova, privacy)
   - Drift detection (ADWIN, statistical tests)
   - Gap: No prior work on federated drift detection for media forensics
   
3. **Method**
   - System architecture (client-server, sketch protocol)
   - Privacy-preserving sketches (histograms + DP)
   - Drift detection ensemble (KS, PSI, JS)
   - Anomaly detection (clustering)
   - Adaptive thresholding
   
4. **Experiments**
   - Setup (datasets, models, baselines)
   - Exp 1-5 (as defined in plan)
   - Ablation studies (privacy levels, detection algorithms, client numbers)
   
5. **Results & Discussion**
   - Detection performance (latency, accuracy)
   - Privacy-utility trade-offs
   - Communication efficiency
   - Scalability analysis
   
6. **Conclusion & Future Work**
   - Summary of contributions
   - Real-world deployment considerations
   - Extensions (continual learning, Byzantine robustness)

### Key Contributions to Highlight
1. **Problem formulation**: Federated drift detection for deepfake forensics
2. **System design**: Privacy-preserving sketch-based monitoring protocol
3. **Algorithms**: Ensemble drift detection + clustering-based anomaly detection
4. **Empirical validation**: Comprehensive experiments with realistic simulation
5. **Open-source release**: Code, models, experimental framework

### Timeline to Submission
- **Weeks 1-3**: Implementation (core system + simulation)
- **Weeks 4-5**: Experiments (run all 5 experiments, collect results)
- **Weeks 6-7**: Paper writing (first draft)
- **Week 8**: Internal review, revisions
- **Week 9**: Final polish, submission

**Target submission date**: ACM MM 2026 (deadline typically early April) or WACV 2027 (deadline ~August)

---

## Summary of All Decisions

| Aspect | Decision |
|--------|----------|
| Privacy mechanism | Histograms (20 bins) + DP (e=1.0) |
| Drift detection | Ensemble: KS-test + PSI + JS-divergence |
| Baseline | Static global, from training data |
| Anomaly detection | DBSCAN clustering on JS-divergence |
| Thresholding | Global, updated on drift, FPR=0.01 |
| Communication | Push-based, asynchronous, N=500 |
| Data distribution | Non-IID (clustered), power-law sizes |
| Evaluation | Detection latency, precision/recall, false alarms |
| Integration | Wrapper around existing system |
| Target venue | ACM MM 2026 / WACV 2027 |

---

## Next Steps

1. ? Design decisions documented
2. ?? Literature survey (identify related papers, find exact gaps)
3. ?? Detailed implementation roadmap (break down into tasks)
4. ? Prototype implementation
5. ? Experiments
6. ? Paper writing

