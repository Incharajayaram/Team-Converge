# Federated Drift Detection - Detailed Implementation Roadmap

This document provides a task-by-task breakdown for implementing the federated privacy-preserving drift detection system.

---

## Overview

**Total Timeline**: 9 weeks (to paper submission)  
**Workload**: ~20-30 hours/week (manageable for research project)  
**Deliverables**: Working system + experiments + paper draft

---

## Phase 1: Core Implementation (Weeks 1-2)

### Week 1: Foundation Components

#### Task 1.1: Privacy Utilities Module
**File**: `federated/privacy_utils.py`  
**Duration**: 2-3 hours  
**Dependencies**: None

**Subtasks**:
- [ ] Implement Laplace mechanism for DP
  ```python
  def add_laplace_noise(value, sensitivity, epsilon):
      noise = np.random.laplace(0, sensitivity/epsilon)
      return value + noise
  ```
- [ ] Implement Gaussian mechanism for (ε, δ)-DP
- [ ] Privacy budget tracker (cumulative ε)
- [ ] Unit tests for noise addition
- [ ] Validate DP guarantees with simple examples

**Success Criteria**: 
- Noise correctly added with specified ε
- Verify DP composition (sequential queries)

---

#### Task 1.2: Sketch Algorithms Module
**File**: `federated/sketch_algorithms.py`  
**Duration**: 4-5 hours  
**Dependencies**: Task 1.1 (for DP integration)

**Subtasks**:
- [ ] Implement `ScoreHistogram` class
  - 20 bins, range [0, 1]
  - Update method (add new score)
  - Serialize to dict (for JSON transmission)
  - Add DP noise to counts
  
- [ ] Implement `StatisticalSummary` class
  - Running mean (Welford's algorithm)
  - Running std
  - Quantile tracking (P² algorithm or simple buffer)
  
- [ ] Sparse histogram representation (compression)
  ```python
  def to_sparse(self):
      return {i: count for i, count in enumerate(self.bins) if count > 0}
  ```

- [ ] Unit tests
  - Verify histogram binning correct
  - Check sparse compression ratio
  - Validate statistics accuracy

**Success Criteria**:
- Histogram correctly bins scores
- Statistics match numpy reference
- Compression reduces size by ~50%

---

#### Task 1.3: Client Monitor
**File**: `federated/client_monitor.py`  
**Duration**: 5-6 hours  
**Dependencies**: Tasks 1.1, 1.2

**Subtasks**:
- [ ] Implement `ClientMonitor` class
  - Sliding window buffer (last N predictions)
  - Update method (add score, confidence, flags)
  - Generate sketch (histogram + stats + DP noise)
  - Local drift detection (KS-test against baseline)
  
- [ ] Configuration management
  - Window size (default 500)
  - Privacy epsilon (default 1.0)
  - Number of bins (default 20)
  
- [ ] Tracking additional metrics
  - Abstain rate
  - OOD detection rate
  - Confidence distribution

- [ ] Unit tests
  - Verify window sliding
  - Check sketch generation
  - Test DP noise application

**Success Criteria**:
- Monitor correctly accumulates predictions
- Sketch includes histogram + stats
- DP noise applied with specified ε

---

#### Task 1.4: Drift Detection Algorithms
**File**: `federated/drift_detection.py`  
**Duration**: 4-5 hours  
**Dependencies**: Task 1.2

**Subtasks**:
- [ ] Implement KS-test detector
  ```python
  from scipy.stats import ks_2samp
  def ks_drift_test(baseline_hist, current_hist, threshold=0.01):
      # Reconstruct approximate samples from histograms
      # Run KS-test
      # Return drift_detected, p_value
  ```

- [ ] Implement PSI detector
  ```python
  def psi_drift_test(baseline_hist, current_hist, threshold=0.1):
      psi = sum((curr - base) * np.log(curr / base))
      return psi > threshold, psi
  ```

- [ ] Implement JS-divergence detector
  ```python
  from scipy.spatial.distance import jensenshannon
  def js_drift_test(baseline_hist, current_hist, threshold=0.1):
      js_div = jensenshannon(baseline_hist, current_hist)
      return js_div > threshold, js_div
  ```

- [ ] Ensemble detector (2/3 agreement)
  ```python
  def ensemble_drift_detection(baseline, current):
      detectors = [ks_drift_test, psi_drift_test, js_drift_test]
      results = [detector(baseline, current)[0] for detector in detectors]
      return sum(results) >= 2  # Majority vote
  ```

- [ ] Unit tests with synthetic data

**Success Criteria**:
- Each detector correctly identifies known distribution shifts
- Ensemble reduces false positives vs single detector

---

### Week 2: Server Components

#### Task 2.1: Drift Server
**File**: `federated/drift_server.py`  
**Duration**: 6-7 hours  
**Dependencies**: Tasks 1.2, 1.4

**Subtasks**:
- [ ] Implement `DriftServer` class
  - Store baseline distribution
  - Receive sketches from clients
  - Aggregate histograms (weighted average)
  - Detect global drift (using ensemble)
  - Track drift history over rounds
  
- [ ] Aggregation methods
  ```python
  def aggregate_distributions(self, client_histograms):
      # Weighted average (by number of samples)
      weights = [hist['num_samples'] for hist in client_histograms]
      aggregated = np.average(
          [hist['bins'] for hist in client_histograms],
          axis=0, weights=weights
      )
      return aggregated / aggregated.sum()  # Normalize
  ```

- [ ] Drift reporting
  ```python
  def get_drift_report(self):
      return {
          'drift_detected': self.current_drift_status,
          'drift_scores': {
              'ks': self.ks_score,
              'psi': self.psi_score,
              'js': self.js_score
          },
          'affected_clients': self.anomalous_clients
      }
  ```

- [ ] Persistence (save/load state)

- [ ] Unit tests

**Success Criteria**:
- Server correctly aggregates client histograms
- Drift detection works on aggregated distribution
- Drift reports are informative

---

#### Task 2.2: Anomaly Detection (Client Clustering)
**File**: `federated/anomaly_detection.py`  
**Duration**: 4-5 hours  
**Dependencies**: Task 2.1

**Subtasks**:
- [ ] Implement pairwise divergence matrix
  ```python
  def compute_divergence_matrix(client_histograms):
      n = len(client_histograms)
      matrix = np.zeros((n, n))
      for i in range(n):
          for j in range(i+1, n):
              div = jensenshannon(client_histograms[i], client_histograms[j])
              matrix[i, j] = matrix[j, i] = div
      return matrix
  ```

- [ ] Implement DBSCAN clustering
  ```python
  from sklearn.cluster import DBSCAN
  def detect_anomalous_clients(divergence_matrix, eps=0.15, min_samples=2):
      clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
      labels = clustering.fit_predict(divergence_matrix)
      
      # Find majority cluster
      label_counts = Counter(labels[labels != -1])
      majority_label = label_counts.most_common(1)[0][0]
      
      # Anomalous = not in majority cluster
      anomalous = [i for i, label in enumerate(labels) 
                   if label != majority_label and label != -1]
      return anomalous
  ```

- [ ] Unit tests with synthetic scenarios

**Success Criteria**:
- Correctly identifies clients with divergent distributions
- Handles edge cases (all clients similar, single outlier)

---

#### Task 2.3: Adaptive Threshold Manager
**File**: `federated/adaptive_threshold.py`  
**Duration**: 3-4 hours  
**Dependencies**: Task 2.1

**Subtasks**:
- [ ] Implement threshold calibration
  ```python
  def calibrate_threshold(aggregated_hist, target_fpr=0.01):
      # Reconstruct CDF from histogram
      bin_centers = np.linspace(0.025, 0.975, 20)
      cdf = np.cumsum(aggregated_hist) / np.sum(aggregated_hist)
      
      # Find threshold for target FPR
      threshold_idx = np.searchsorted(cdf, 1 - target_fpr)
      threshold = bin_centers[threshold_idx]
      return threshold
  ```

- [ ] Implement threshold update protocol
  - Trigger: Only update when drift detected
  - Broadcast: Send to all clients
  - Logging: Track threshold changes over time

- [ ] Unit tests

**Success Criteria**:
- Threshold achieves approximately target FPR
- Updates only when drift detected

---

#### Task 2.4: Communication Layer (Server Side)
**File**: `federated/server_api.py`  
**Duration**: 3-4 hours  
**Dependencies**: Tasks 2.1, 2.2, 2.3

**Subtasks**:
- [ ] Implement Flask REST API
  ```python
  from flask import Flask, request, jsonify
  
  @app.route('/submit_sketch', methods=['POST'])
  def submit_sketch():
      data = request.json
      client_id = data['client_id']
      histogram = decompress_histogram(data['histogram'])
      stats = data['stats']
      
      drift_server.receive_sketch(client_id, histogram, stats)
      
      # Check if drift detected
      if drift_server.drift_detected:
          new_threshold = threshold_manager.calibrate()
          return jsonify({
              'status': 'received',
              'threshold_update': new_threshold
          })
      
      return jsonify({'status': 'received'})
  
  @app.route('/get_status', methods=['GET'])
  def get_status():
      return jsonify(drift_server.get_drift_report())
  ```

- [ ] Implement sketch decompression
- [ ] Add logging for all requests
- [ ] Error handling

**Success Criteria**:
- Server receives and processes client sketches
- API returns correct responses

---

## Phase 2: Simulation Framework (Weeks 2-3)

### Week 2-3: Simulation Environment

#### Task 3.1: Federated Client Wrapper
**File**: `federated/federated_client.py`  
**Duration**: 3-4 hours  
**Dependencies**: Tasks 1.3, 2.4

**Subtasks**:
- [ ] Implement `FederatedClient` class
  - Wraps ClientMonitor
  - Handles communication with server (REST API)
  - Triggers sketch sending after N predictions
  - Receives and applies threshold updates
  
- [ ] Integration with existing inference pipeline
  ```python
  class FederatedDeepfakeDetector:
      def __init__(self, model, server_url, client_id):
          self.model = model
          self.fed_client = FederatedClient(server_url, client_id)
      
      def detect(self, image):
          score = self.model(image)
          self.fed_client.on_prediction(score)
          return score > self.fed_client.threshold
  ```

- [ ] Unit tests (mock server)

**Success Criteria**:
- Client correctly sends sketches to server
- Client receives and applies threshold updates

---

#### Task 3.2: Data Partitioning (Non-IID)
**File**: `federated/simulation/data_partitioning.py`  
**Duration**: 5-6 hours  
**Dependencies**: None

**Subtasks**:
- [ ] Implement clustering-based non-IID split
  - Extract features using pretrained model (ResNet)
  - K-means clustering (num_clusters = num_clients * 2)
  - Assign clusters to clients using Dirichlet distribution
  
- [ ] Implement power-law data sizes
  - Zipf distribution for client sample counts
  - Ensure minimum samples per client (50)
  
- [ ] Validation
  - Visualize data distribution (t-SNE)
  - Compute heterogeneity metrics (KL-divergence between clients)

- [ ] Unit tests

**Success Criteria**:
- Non-IID splits have clear distribution differences
- Power-law distribution matches expected shape

---

#### Task 3.3: Drift Scenario Injection
**File**: `federated/simulation/drift_scenarios.py`  
**Duration**: 6-7 hours  
**Dependencies**: Task 3.2

**Subtasks**:
- [ ] Implement attack injection using ECDD edge cases
  - Blur attack (GaussianBlur radius 4)
  - JPEG compression (quality 30)
  - Resize artifacts (downsample + upsample)
  - Multi-face injection (from ECDD data)
  
- [ ] Implement drift patterns
  - **Sudden drift**: Inject 30% attacked samples at specific round
  - **Gradual drift**: Linearly increase attack proportion over 50 rounds
  - **Localized drift**: Only affect subset of clients (30%)
  - **Correlated drift**: Multiple clients see same attack simultaneously
  
- [ ] Configurable intensity and timing
  ```python
  class DriftScenario:
      def __init__(self, attack_type, intensity, start_round, affected_clients):
          self.attack_type = attack_type
          self.intensity = intensity
          self.start_round = start_round
          self.affected_clients = affected_clients
      
      def apply(self, round_num, client_id, data):
          if (round_num >= self.start_round and 
              client_id in self.affected_clients):
              return inject_attack(data, self.attack_type, self.intensity)
          return data
  ```

- [ ] Unit tests (verify transformations applied correctly)

**Success Criteria**:
- Attacks correctly applied to images
- Drift patterns match specifications
- Visual inspection confirms realistic transformations

---

#### Task 3.4: Simulation Orchestrator
**File**: `federated/simulation/fed_drift_simulator.py`  
**Duration**: 7-8 hours  
**Dependencies**: Tasks 3.1, 3.2, 3.3

**Subtasks**:
- [ ] Implement `FederatedDriftSimulator` class
  - Create clients with non-IID data
  - Initialize server
  - Run rounds (clients process data → send sketches → server aggregates)
  - Inject drift according to scenario
  - Simulate client dropout (20% per round)
  
- [ ] Implement round execution
  ```python
  def run_round(self, round_num):
      # Select active clients (80%)
      active_clients = self.sample_clients(dropout_rate=0.2)
      
      # Each client processes local data
      for client_id in active_clients:
          client = self.clients[client_id]
          data = self.get_client_data(client_id, round_num)
          
          # Apply drift if applicable
          if self.scenario:
              data = self.scenario.apply(round_num, client_id, data)
          
          # Client inference + monitoring
          for image, label in data:
              score = client.detect(image)
          
          # Client sends sketch if window full
          client.maybe_send_sketch()
      
      # Server aggregates and checks drift
      drift_detected = self.server.check_drift()
      
      # Log metrics
      self.log_round_metrics(round_num, drift_detected)
  ```

- [ ] Logging and checkpointing
  - Save state every 10 rounds
  - Log drift scores, threshold updates, anomalous clients
  
- [ ] Progress tracking (tqdm)

**Success Criteria**:
- Simulation runs end-to-end without errors
- Correct number of clients and rounds executed
- Logs capture all relevant metrics

---

#### Task 3.5: Evaluation Metrics Module
**File**: `federated/simulation/evaluation_metrics.py`  
**Duration**: 4-5 hours  
**Dependencies**: Task 3.4

**Subtasks**:
- [ ] Implement `ExperimentMetrics` class (from design doc)
  - Detection latency
  - Client identification precision/recall/F1
  - False alarm rate
  - Communication overhead (bytes transmitted)
  
- [ ] Implement comparison utilities
  - Compare federated vs baselines (centralized, isolated)
  - Statistical significance testing (paired t-test)
  - Generate comparison tables
  
- [ ] Implement plotting utilities
  ```python
  def plot_detection_latency(results):
      # Bar chart: federated vs baselines
      
  def plot_privacy_utility_tradeoff(results):
      # Line chart: epsilon vs detection accuracy
      
  def plot_scalability(results):
      # Line chart: num_clients vs aggregation_time
  ```

- [ ] Unit tests

**Success Criteria**:
- Metrics correctly computed from simulation logs
- Plots are clear and publication-ready

---

## Phase 3: Experiments (Weeks 4-5)

### Week 4: Core Experiments

#### Task 4.1: Experiment 1 - Baseline (No Drift)
**Duration**: 4 hours  
**Dependencies**: Phase 2 complete

**Subtasks**:
- [ ] Setup: 10 clients, CelebDF test set, no drift injection
- [ ] Run 10 times with different random seeds
- [ ] Measure: False alarm rate, communication overhead
- [ ] Generate results JSON

**Success Criteria**:
- False alarm rate < 5%
- Consistent results across runs

---

#### Task 4.2: Experiment 2 - Sudden Attack Emergence
**Duration**: 5 hours  
**Dependencies**: Task 4.1

**Subtasks**:
- [ ] Setup: 10 clients, inject blur attack at round 50 to 3 clients (30%)
- [ ] Run 10 times
- [ ] Measure: Detection latency, client identification precision/recall
- [ ] Compare: Federated vs centralized vs isolated
- [ ] Generate plots and tables

**Success Criteria**:
- Federated detects drift within 10 rounds (90%+ of runs)
- Client identification F1 > 0.7

---

#### Task 4.3: Experiment 3 - Gradual Distribution Shift
**Duration**: 5 hours  
**Dependencies**: Task 4.2

**Subtasks**:
- [ ] Setup: 20 clients, gradually increase JPEG compression over 50 rounds
- [ ] Run 10 times
- [ ] Measure: Detection timing, drift magnitude estimation
- [ ] Visualize drift scores over time (line plot)

**Success Criteria**:
- Drift detected before attack reaches 50% intensity
- PSI effectively tracks gradual shift

---

#### Task 4.4: Experiment 4 - Privacy-Utility Trade-off
**Duration**: 6 hours  
**Dependencies**: Task 4.1

**Subtasks**:
- [ ] Setup: Vary epsilon (0.1, 0.5, 1.0, 5.0, 10.0, ∞)
- [ ] Run sudden attack scenario (Exp 2) with each epsilon
- [ ] Run 10 times per epsilon
- [ ] Measure: Detection accuracy, latency, false alarms
- [ ] Plot: Privacy (epsilon) vs Detection F1

**Success Criteria**:
- Clear trend: higher epsilon → better detection
- ε=1.0 provides good balance (F1 > 0.75)

---

### Week 5: Advanced Experiments

#### Task 4.5: Experiment 5 - Scalability
**Duration**: 6 hours  
**Dependencies**: Task 4.1

**Subtasks**:
- [ ] Setup: Vary num_clients (10, 20, 50, 100, 200)
- [ ] Run sudden attack scenario with each configuration
- [ ] Run 5 times per configuration (fewer runs due to compute)
- [ ] Measure: Aggregation time, communication bytes, detection accuracy
- [ ] Plot: Scalability curves

**Success Criteria**:
- Aggregation time scales linearly (< 1 sec for 100 clients)
- Detection accuracy stable across client counts

---

#### Task 4.6: Ablation Studies
**Duration**: 5 hours  
**Dependencies**: All experiments

**Subtasks**:
- [ ] Ablate ensemble components (only KS, only PSI, only JS vs ensemble)
- [ ] Ablate anomaly detection (with/without DBSCAN clustering)
- [ ] Ablate adaptive thresholding (fixed vs adaptive)
- [ ] Run each ablation 5 times
- [ ] Generate ablation table

**Success Criteria**:
- Ensemble outperforms single detectors
- Clustering improves client identification
- Adaptive thresholding reduces false alarms

---

#### Task 4.7: Results Compilation
**Duration**: 4 hours  
**Dependencies**: All experiments

**Subtasks**:
- [ ] Aggregate all results into master spreadsheet
- [ ] Generate all plots (high-resolution, publication-ready)
- [ ] Create comparison tables (LaTeX format)
- [ ] Statistical significance tests
- [ ] Write results summary document

**Success Criteria**:
- All figures and tables ready for paper
- Clear narrative from results

---

## Phase 4: Paper Writing (Weeks 6-7)

### Week 6: Draft Writing

#### Task 5.1: Introduction & Related Work
**Duration**: 6-8 hours  

**Subtasks**:
- [ ] Introduction (2 pages)
  - Motivation: Deepfakes evolving, privacy concerns
  - Challenge: Collaborative monitoring without data sharing
  - Solution preview: Federated drift detection
  - Contributions list (4-5 bullet points)
  
- [ ] Related Work (2-3 pages)
  - Section 2.1: Deepfake Detection
  - Section 2.2: Federated Learning
  - Section 2.3: Drift Detection
  - Section 2.4: Privacy-Preserving Monitoring
  - Section 2.5: Positioning (research gap)

**Success Criteria**:
- Clear motivation and problem statement
- Comprehensive but focused related work
- Research gap explicitly stated

---

#### Task 5.2: Method Section
**Duration**: 8-10 hours  

**Subtasks**:
- [ ] Section 3.1: System Overview (architecture diagram)
- [ ] Section 3.2: Privacy-Preserving Sketches (histogram + DP)
- [ ] Section 3.3: Drift Detection Ensemble (KS, PSI, JS)
- [ ] Section 3.4: Anomaly Detection (clustering)
- [ ] Section 3.5: Adaptive Thresholding
- [ ] Section 3.6: Communication Protocol

**Success Criteria**:
- Clear technical description
- Algorithms presented formally (pseudocode or equations)
- Reproducible from description

---

#### Task 5.3: Experiments Section
**Duration**: 6-8 hours  

**Subtasks**:
- [ ] Section 4.1: Experimental Setup
  - Datasets (CelebDF)
  - Models (LaDeDa teacher/student)
  - Baselines
  - Metrics
  
- [ ] Section 4.2: Results
  - Exp 1: Baseline
  - Exp 2: Sudden attack
  - Exp 3: Gradual shift
  - Exp 4: Privacy trade-off
  - Exp 5: Scalability
  
- [ ] Section 4.3: Ablation Studies
- [ ] Section 4.4: Discussion

**Success Criteria**:
- All experiments clearly described
- Results presented with figures and tables
- Insights and analysis provided

---

### Week 7: Polishing

#### Task 5.4: Conclusion & Abstract
**Duration**: 3-4 hours  

**Subtasks**:
- [ ] Conclusion (1 page)
  - Summary of contributions
  - Limitations
  - Future work
  
- [ ] Abstract (last to write, 150-250 words)
  - Problem, method, results, impact

**Success Criteria**:
- Strong conclusion
- Compelling abstract

---

#### Task 5.5: Figures & Tables
**Duration**: 4-5 hours  

**Subtasks**:
- [ ] Polish all figures (consistent style, fonts, colors)
- [ ] Format all tables (LaTeX)
- [ ] Add captions with detailed descriptions
- [ ] Ensure all referenced in text

**Success Criteria**:
- Professional quality figures
- Self-explanatory captions

---

#### Task 5.6: Formatting & References
**Duration**: 3-4 hours  

**Subtasks**:
- [ ] Format according to venue template (ACM MM / WACV)
- [ ] Complete bibliography (40-60 references)
- [ ] Check all citations in text
- [ ] Proofread for grammar and typos
- [ ] Check page limit compliance

**Success Criteria**:
- Properly formatted
- All references complete and consistent

---

## Phase 5: Review & Submission (Weeks 8-9)

### Week 8: Internal Review

#### Task 6.1: Self-Review
**Duration**: 4-5 hours  

**Subtasks**:
- [ ] Read paper end-to-end as if reviewer
- [ ] Check logical flow
- [ ] Verify all claims supported by results
- [ ] Identify weak points
- [ ] Revise accordingly

---

#### Task 6.2: Peer Feedback
**Duration**: Variable (depends on collaborators)  

**Subtasks**:
- [ ] Share with advisor/collaborators
- [ ] Collect feedback
- [ ] Prioritize revisions
- [ ] Implement major revisions

---

### Week 9: Final Polish & Submission

#### Task 6.3: Final Revisions
**Duration**: 6-8 hours  

**Subtasks**:
- [ ] Address all feedback
- [ ] Final proofread
- [ ] Supplementary materials (code, data)
- [ ] Submission checklist

---

#### Task 6.4: Submission
**Duration**: 2-3 hours  

**Subtasks**:
- [ ] Create camera-ready PDF
- [ ] Write submission abstract (for conference system)
- [ ] Upload all files
- [ ] Double-check requirements
- [ ] Submit!

---

## Task Dependencies Graph

```
Phase 1 (Foundation)
├── 1.1 (Privacy) ────────────┐
├── 1.2 (Sketches) ←─────────┤
├── 1.3 (Client Monitor) ←───┼────┐
├── 1.4 (Drift Detection) ←──┤    │
├── 2.1 (Drift Server) ←─────┴──┐ │
├── 2.2 (Anomaly) ←─────────────┤ │
├── 2.3 (Threshold) ←───────────┤ │
└── 2.4 (Server API) ←──────────┴─┤
                                   │
Phase 2 (Simulation)               │
├── 3.1 (Fed Client) ←─────────────┤
├── 3.2 (Data Partition)           │
├── 3.3 (Drift Scenarios)          │
├── 3.4 (Simulator) ←──────────────┴─ All above
└── 3.5 (Metrics) ←────────────────── 3.4
                                      │
Phase 3 (Experiments)                 │
├── 4.1-4.5 (Experiments) ←───────────┘
├── 4.6 (Ablations) ←──────────────── 4.1-4.5
└── 4.7 (Compilation) ←──────────────── All above
                                        │
Phase 4 (Paper)                         │
├── 5.1 (Intro + Related) ←─────────────┤
├── 5.2 (Method) ←──────────────────────┤
├── 5.3 (Experiments) ←─────────────────┘
├── 5.4 (Conclusion + Abstract) ←─────── 5.1-5.3
├── 5.5 (Figures) ←──────────────────────┴─ 4.7
└── 5.6 (Formatting)
                                          │
Phase 5 (Submission)                      │
├── 6.1 (Self-Review) ←───────────────────┘
├── 6.2 (Peer Feedback)
├── 6.3 (Final Revisions)
└── 6.4 (Submit!)
```

---

## Resource Requirements

### Compute
- **Training**: Not needed (use existing trained models)
- **Simulation**: 1 GPU (or CPU, slower) for inference
  - Estimated: 2-4 hours per experiment on GPU
  - Total: ~30-40 GPU hours for all experiments
- **Analysis**: CPU only (lightweight)

### Storage
- **Code**: < 10 MB
- **Datasets**: ~10 GB (CelebDF already downloaded)
- **Results**: ~1 GB (logs, checkpoints, plots)
- **Total**: ~11 GB

### Dependencies (Python packages)
```
torch, torchvision (already have)
numpy, scipy (standard)
scikit-learn (clustering, metrics)
flask (REST API)
matplotlib, seaborn (plotting)
tqdm (progress bars)
pytest (testing)
```

---

## Risk Mitigation

### Risk 1: Experiments don't show clear results
**Mitigation**: 
- Start with strong baseline (your trained models are good)
- Use clear drift scenarios (ECDD edge cases are realistic)
- If results weak, adjust scenarios or metrics

### Risk 2: Implementation takes longer than expected
**Mitigation**:
- Prioritize core functionality (MVP first)
- Skip optional features (Byzantine robustness, real deployment demo)
- Reduce experiment scope if needed (fewer runs, fewer conditions)

### Risk 3: Paper rejected
**Mitigation**:
- Target realistic venues (ACM MM, WACV, not CVPR)
- Get feedback early from advisor
- Have backup submission plan (workshops, lower-tier conferences)

---

## Checklist Summary

### Implementation Milestones
- [ ] **Week 2 End**: Core components working (client monitor + drift server)
- [ ] **Week 3 End**: Simulation framework complete
- [ ] **Week 5 End**: All experiments done, results compiled
- [ ] **Week 7 End**: Paper draft complete
- [ ] **Week 9 End**: Submitted!

### Quality Gates
- [ ] All components have unit tests
- [ ] End-to-end simulation runs without errors
- [ ] At least 1 experiment shows clear positive results before proceeding to paper
- [ ] Paper reviewed by at least 1 other person before submission

---

## Next Steps

Now we have:
1. ✅ Complete design decisions
2. ✅ Literature survey
3. ✅ Detailed implementation roadmap

**Ready to start coding?** 

I recommend beginning with **Task 1.1 (Privacy Utils)** - it's the foundation and takes only 2-3 hours. Would you like me to implement that first module now?

