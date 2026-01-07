# Implementation Sequence - Federated Drift Detection with Teacher-Student Architecture

This document lists the **exact sequence** of tasks to implement, incorporating the teacher-student distillation paradigm into federated learning.

---

## 🎯 Enhanced Design: Hierarchical Federated Learning

### Key Modification to Original Plan
We integrate your existing **teacher-student architecture** into the federated setting:

- **Level 1 (Edge)**: Student models on Arduino Nicla devices (lightweight clients)
- **Level 2 (Hub)**: Teacher models on Raspberry Pi (local aggregators + teachers)
- **Level 3 (Central)**: Global server (aggregates teachers + coordinates drift detection)

### Benefits
1. **Realistic deployment**: Matches actual edge infrastructure
2. **Communication efficiency**: Students → nearby Pi (not central server)
3. **Heterogeneous models**: Different architectures (student vs teacher)
4. **Knowledge distillation in FL**: Teachers guide students with soft labels
5. **Novel research contribution**: Hierarchical FL for deepfake detection (very few papers on this!)

---

## 📋 Complete Implementation Sequence

### **PHASE 1: Foundation (Week 1-2)**

#### ✅ Task 1: Privacy Utilities (2-3 hours)
**File**: `core/privacy_utils.py`

**What to implement**:
```python
def add_laplace_noise(value, sensitivity, epsilon):
    """Add Laplace noise for differential privacy"""
    
def add_gaussian_noise(value, sensitivity, epsilon, delta):
    """Add Gaussian noise for (ε, δ)-DP"""
    
class PrivacyBudgetTracker:
    """Track cumulative privacy budget across queries"""
    def __init__(self, total_budget):
    def consume(self, epsilon):
    def remaining(self):
```

**Success criteria**: Unit tests pass, noise addition correct

---

#### ✅ Task 2: Sketch Algorithms (4-5 hours)
**File**: `core/sketch_algorithms.py`

**What to implement**:
```python
class ScoreHistogram:
    """20-bin histogram for score distributions"""
    def __init__(self, num_bins=20, range=(0, 1)):
    def update(self, value):
    def get_normalized(self):
    def to_sparse_dict(self):  # Compression
    def add_dp_noise(self, epsilon):
    
class StatisticalSummary:
    """Running statistics (Welford's algorithm)"""
    def __init__(self):
    def update(self, value):
    def get_stats(self):  # Returns mean, std, quantiles
```

**Success criteria**: Histogram bins correctly, stats match numpy, compression works

---

#### ✅ Task 3: Drift Detection Algorithms (4-5 hours)
**File**: `core/drift_detection.py`

**What to implement**:
```python
def ks_drift_test(baseline_hist, current_hist, threshold=0.01):
    """Kolmogorov-Smirnov test for sudden drift"""
    
def psi_drift_test(baseline_hist, current_hist, threshold=0.1):
    """Population Stability Index for gradual drift"""
    
def js_drift_test(baseline_hist, current_hist, threshold=0.1):
    """Jensen-Shannon divergence for distribution similarity"""
    
class EnsembleDriftDetector:
    """Ensemble of KS, PSI, JS with majority voting"""
    def __init__(self, ks_threshold=0.01, psi_threshold=0.1, js_threshold=0.1):
    def detect(self, baseline_hist, current_hist):
        # Returns: drift_detected (bool), scores (dict)
```

**Success criteria**: Each detector identifies synthetic drift, ensemble reduces false positives

---

#### ✅ Task 4: Anomaly Detection (4-5 hours)
**File**: `core/anomaly_detection.py`

**What to implement**:
```python
def compute_divergence_matrix(client_histograms):
    """Pairwise JS-divergence between all clients"""
    
def detect_anomalous_clients(divergence_matrix, eps=0.15, min_samples=2):
    """DBSCAN clustering to find anomalous clients"""
    # Returns: list of anomalous client IDs
    
class ClientClusterer:
    """Wrapper for client clustering and analysis"""
    def __init__(self, eps=0.15, min_samples=2):
    def fit(self, client_histograms):
    def get_anomalous_clients(self):
    def get_clusters(self):
```

**Success criteria**: Correctly identifies clients with divergent distributions

---

#### ✅ Task 5: Client Monitor (5-6 hours)
**File**: `client/client_monitor.py`

**What to implement**:
```python
class ClientMonitor:
    """Local monitoring for a single client"""
    def __init__(self, window_size=500, num_bins=20, epsilon=1.0):
        self.buffer = deque(maxlen=window_size)
        self.histogram = ScoreHistogram(num_bins)
        self.stats = StatisticalSummary()
        
    def update(self, score, confidence, is_ood=False, abstained=False):
        """Add new prediction to monitor"""
        
    def get_sketch(self, apply_dp=True):
        """Generate privacy-preserving sketch"""
        # Returns: {
        #     'histogram': sparse dict,
        #     'stats': {mean, std, quantiles},
        #     'metadata': {num_samples, abstain_rate, ood_rate}
        # }
        
    def detect_local_drift(self, baseline_hist):
        """Optional: Local drift detection before sending"""
```

**Success criteria**: Monitor accumulates predictions, generates sketches, applies DP

---

#### ✅ Task 6: Student Client (NEW - 3-4 hours)
**File**: `client/student_client.py`

**What to implement**:
```python
class StudentClient:
    """Client running student model with federated monitoring"""
    def __init__(self, student_model, hub_url, client_id):
        self.model = student_model  # Your tiny LaDeDa
        self.monitor = ClientMonitor()
        self.hub_url = hub_url  # Send to local Pi, not central server
        self.threshold = 0.5
        
    def predict(self, image):
        """Run inference + update monitor"""
        score = self.model(image)
        confidence = compute_confidence(score)  # From your existing code
        self.monitor.update(score, confidence)
        
        # Check if should send sketch to hub
        if self.monitor.buffer_full():
            self.send_sketch_to_hub()
        
        return score > self.threshold
    
    def send_sketch_to_hub(self):
        """Send sketch to local Pi hub"""
        sketch = self.monitor.get_sketch()
        response = requests.post(f"{self.hub_url}/submit_sketch", json=sketch)
        
        # Update threshold if hub sends new one
        if 'threshold_update' in response.json():
            self.threshold = response.json()['threshold_update']
```

**Success criteria**: Student model runs, monitor updates, communicates with hub

---

#### ✅ Task 7: Teacher Hub Aggregator (NEW - 5-6 hours)
**File**: `server/teacher_aggregator.py`

**What to implement**:
```python
class TeacherHubAggregator:
    """Raspberry Pi running teacher model + local aggregation"""
    def __init__(self, teacher_model, central_server_url, hub_id):
        self.model = teacher_model  # Your LaDeDa teacher
        self.student_clients = {}  # Track connected students
        self.local_aggregated_hist = None
        self.central_server_url = central_server_url
        
    def receive_student_sketch(self, student_id, sketch):
        """Receive sketch from student device"""
        self.student_clients[student_id] = sketch
        
        # Aggregate local students
        if len(self.student_clients) >= self.min_students:
            self.aggregate_local_students()
    
    def aggregate_local_students(self):
        """Aggregate sketches from students in this hub"""
        student_hists = [s['histogram'] for s in self.student_clients.values()]
        weights = [s['metadata']['num_samples'] for s in self.student_clients.values()]
        
        self.local_aggregated_hist = np.average(student_hists, weights=weights)
        
        # Send aggregated sketch to central server
        self.send_to_central_server()
    
    def send_to_central_server(self):
        """Send aggregated sketch to central server"""
        payload = {
            'hub_id': self.hub_id,
            'aggregated_histogram': self.local_aggregated_hist.tolist(),
            'num_students': len(self.student_clients),
            'hub_stats': self.compute_hub_stats()
        }
        response = requests.post(f"{self.central_server_url}/submit_hub_aggregation", json=payload)
        
        # Broadcast threshold updates to students
        if 'threshold_update' in response.json():
            self.broadcast_threshold_to_students(response.json()['threshold_update'])
    
    def broadcast_threshold_to_students(self, new_threshold):
        """Send updated threshold to all connected students"""
        for student_id in self.student_clients:
            # In real deployment, send via network
            # In simulation, direct call
            pass
```

**Success criteria**: Hub aggregates student sketches, communicates with central server

---

#### ✅ Task 8: Central Drift Server (6-7 hours)
**File**: `server/drift_server.py`

**What to implement**:
```python
class CentralDriftServer:
    """Central server coordinating all hubs"""
    def __init__(self, baseline_dist, num_hubs):
        self.baseline = baseline_dist
        self.hubs = {}  # Track all hubs
        self.drift_detector = EnsembleDriftDetector()
        self.anomaly_detector = ClientClusterer()
        self.global_aggregated_hist = None
        
    def receive_hub_aggregation(self, hub_id, aggregated_hist, metadata):
        """Receive aggregated sketch from hub"""
        self.hubs[hub_id] = {
            'histogram': aggregated_hist,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        # Aggregate all hubs
        if len(self.hubs) >= self.min_hubs_for_aggregation:
            self.aggregate_global()
    
    def aggregate_global(self):
        """Aggregate all hub distributions"""
        hub_hists = [h['histogram'] for h in self.hubs.values()]
        weights = [h['metadata']['num_students'] for h in self.hubs.values()]
        
        self.global_aggregated_hist = np.average(hub_hists, weights=weights)
        
        # Detect drift
        self.check_global_drift()
        
        # Detect anomalous hubs
        self.check_anomalous_hubs()
    
    def check_global_drift(self):
        """Check if global distribution has drifted"""
        drift_detected, scores = self.drift_detector.detect(
            self.baseline, 
            self.global_aggregated_hist
        )
        
        if drift_detected:
            self.handle_drift_detected(scores)
    
    def check_anomalous_hubs(self):
        """Identify hubs with divergent distributions"""
        hub_hists = [h['histogram'] for h in self.hubs.values()]
        anomalous = self.anomaly_detector.detect_anomalous_clients(hub_hists)
        
        if anomalous:
            self.handle_anomalous_hubs(anomalous)
    
    def handle_drift_detected(self, scores):
        """Handle drift detection - recalibrate threshold"""
        new_threshold = self.calibrate_threshold()
        self.broadcast_threshold_to_hubs(new_threshold)
        
        # Log for analysis
        self.log_drift_event(scores, new_threshold)
    
    def get_drift_report(self):
        """Generate comprehensive drift report"""
        return {
            'drift_detected': self.drift_detected,
            'drift_scores': self.drift_scores,
            'anomalous_hubs': self.anomalous_hubs,
            'global_distribution': self.global_aggregated_hist.tolist(),
            'timestamp': time.time()
        }
```

**Success criteria**: Server aggregates hubs, detects drift, coordinates system

---

#### ✅ Task 9: Adaptive Threshold Manager (3-4 hours)
**File**: `server/adaptive_threshold.py`

**What to implement**:
```python
class AdaptiveThresholdManager:
    """Federated threshold calibration"""
    def __init__(self, initial_threshold=0.5, target_fpr=0.01):
        self.threshold = initial_threshold
        self.target_fpr = target_fpr
        self.threshold_history = []
        
    def calibrate_threshold(self, aggregated_hist):
        """Compute optimal threshold from aggregated distribution"""
        # Reconstruct CDF
        bin_centers = np.linspace(0.025, 0.975, 20)
        cdf = np.cumsum(aggregated_hist) / np.sum(aggregated_hist)
        
        # Find threshold for target FPR
        threshold_idx = np.searchsorted(cdf, 1 - self.target_fpr)
        new_threshold = bin_centers[threshold_idx]
        
        self.threshold = new_threshold
        self.threshold_history.append({
            'threshold': new_threshold,
            'timestamp': time.time()
        })
        
        return new_threshold
```

**Success criteria**: Threshold calibration achieves target FPR

---

#### ✅ Task 10: Server REST API (3-4 hours)
**File**: `server/server_api.py`

**What to implement**:
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
drift_server = None  # Initialize in main

@app.route('/submit_hub_aggregation', methods=['POST'])
def submit_hub_aggregation():
    """Receive aggregated sketch from hub"""
    data = request.json
    hub_id = data['hub_id']
    aggregated_hist = np.array(data['aggregated_histogram'])
    metadata = data['hub_stats']
    
    drift_server.receive_hub_aggregation(hub_id, aggregated_hist, metadata)
    
    # Check if threshold updated
    response = {'status': 'received'}
    if drift_server.threshold_updated:
        response['threshold_update'] = drift_server.current_threshold
    
    return jsonify(response)

@app.route('/get_status', methods=['GET'])
def get_status():
    """Get current system status"""
    return jsonify(drift_server.get_drift_report())

@app.route('/get_threshold', methods=['GET'])
def get_threshold():
    """Get current threshold"""
    return jsonify({'threshold': drift_server.current_threshold})
```

**Success criteria**: API receives requests, processes correctly, returns responses

---

### **PHASE 2: Simulation Framework (Week 2-3)**

#### ✅ Task 11: Data Partitioning (5-6 hours)
**File**: `simulation/data_partitioning.py`

**What to implement**:
```python
def create_non_iid_splits(dataset, num_students, concentration=0.5):
    """Create non-IID data splits for student clients"""
    # Use K-means clustering on features
    # Assign clusters using Dirichlet distribution
    
def create_powerlaw_sizes(num_students, total_samples, alpha=1.5):
    """Sample client data sizes from power-law (realistic heterogeneity)"""
    
def assign_students_to_hubs(num_students, num_hubs):
    """Assign student clients to hub devices"""
    # Example: 3-5 students per hub
```

**Success criteria**: Non-IID splits created, visualized, heterogeneous

---

#### ✅ Task 12: Drift Scenario Injection (6-7 hours)
**File**: `simulation/drift_scenarios.py`

**What to implement**:
```python
class DriftScenario:
    """Base class for drift scenarios"""
    def apply(self, round_num, client_id, data):
        """Apply drift transformation to data"""
        
class SuddenAttackScenario(DriftScenario):
    """Sudden injection of new attack type"""
    def __init__(self, attack_type='blur', intensity=0.3, start_round=50, affected_hubs=[0, 1]):
        
class GradualDriftScenario(DriftScenario):
    """Gradual increase in attack intensity"""
    def __init__(self, attack_type='jpeg', intensity_start=0.0, intensity_end=0.5, duration=50):
        
class LocalizedDriftScenario(DriftScenario):
    """Only some hubs affected"""
    
class CorrelatedDriftScenario(DriftScenario):
    """Multiple hubs see same attack simultaneously"""
    
def inject_attack(image, attack_type, intensity):
    """Apply ECDD-style attack transformation"""
    # blur, jpeg compression, resize, multi-face, etc.
```

**Success criteria**: Attacks correctly applied, scenarios configurable

---

#### ✅ Task 13: Hierarchical Federated Simulator (7-8 hours)
**File**: `simulation/fed_drift_simulator.py`

**What to implement**:
```python
class HierarchicalFedSimulator:
    """Simulate hierarchical federated learning (students → hubs → central)"""
    def __init__(self, 
                 student_model_path,
                 teacher_model_path,
                 dataset_path,
                 num_hubs=3,
                 students_per_hub=5,
                 baseline_dist=None):
        
        # Load models
        self.student_model = load_model(student_model_path)
        self.teacher_model = load_model(teacher_model_path)
        
        # Create clients
        self.hubs = self.create_hubs(num_hubs)
        self.students = self.create_students(num_hubs, students_per_hub)
        
        # Central server
        self.central_server = CentralDriftServer(baseline_dist, num_hubs)
        
    def create_hubs(self, num_hubs):
        """Create hub aggregators (Raspberry Pi)"""
        hubs = []
        for i in range(num_hubs):
            hub = TeacherHubAggregator(
                teacher_model=self.teacher_model,
                central_server_url='http://localhost:5000',
                hub_id=i
            )
            hubs.append(hub)
        return hubs
    
    def create_students(self, num_hubs, students_per_hub):
        """Create student clients (Arduino Nicla)"""
        students = []
        for hub_id in range(num_hubs):
            for student_idx in range(students_per_hub):
                student_id = hub_id * students_per_hub + student_idx
                student = StudentClient(
                    student_model=self.student_model,
                    hub_url=f'http://localhost:{6000 + hub_id}',  # Each hub has port
                    client_id=student_id
                )
                students.append(student)
        return students
    
    def run_round(self, round_num, scenario=None):
        """Execute one federated round"""
        # 1. Students process local data
        for student in self.students:
            # Sample local data
            local_data = self.get_student_data(student.client_id)
            
            # Apply drift scenario if applicable
            if scenario:
                local_data = scenario.apply(round_num, student.client_id, local_data)
            
            # Student inference
            for image, label in local_data:
                prediction = student.predict(image)
        
        # 2. Students send sketches to hubs (happens automatically when buffer full)
        
        # 3. Hubs aggregate and send to central server
        for hub in self.hubs:
            if hub.ready_to_aggregate():
                hub.aggregate_local_students()
        
        # 4. Central server checks drift
        if self.central_server.ready_to_aggregate():
            self.central_server.aggregate_global()
            drift_detected = self.central_server.check_global_drift()
            
            if drift_detected:
                # Update thresholds cascade down
                # Central → Hubs → Students
                pass
        
        # 5. Log metrics
        self.log_round_metrics(round_num)
    
    def run_experiment(self, num_rounds=100, scenario=None):
        """Run full experiment"""
        for round_num in tqdm(range(num_rounds)):
            self.run_round(round_num, scenario)
        
        return self.get_results()
```

**Success criteria**: Simulation runs end-to-end, hierarchical communication works

---

#### ✅ Task 14: Evaluation Metrics (4-5 hours)
**File**: `simulation/evaluation_metrics.py`

**What to implement**:
```python
class ExperimentMetrics:
    """Track and compute experiment metrics"""
    def __init__(self, injection_round, affected_clients):
        self.detection_round = None
        self.false_alarms = 0
        self.identified_clients = set()
        
    def update(self, round_num, drift_detected, flagged_clients):
        """Update metrics for current round"""
        
    def compute_final_metrics(self):
        """Compute final metrics"""
        return {
            'detection_latency': self.detection_round - self.injection_round,
            'client_precision': precision,
            'client_recall': recall,
            'client_f1': f1,
            'false_alarm_rate': self.false_alarms / total_rounds
        }

def plot_detection_latency(results):
    """Bar chart comparing methods"""
    
def plot_privacy_utility_tradeoff(results):
    """Line chart: epsilon vs F1"""
    
def plot_scalability(results):
    """Line chart: num_clients vs time"""
    
def generate_comparison_table(results):
    """LaTeX table for paper"""
```

**Success criteria**: Metrics computed correctly, plots publication-ready

---

### **PHASE 3: Experiments (Week 4-5)**

#### ✅ Task 15: Experiment 1 - Baseline (4 hours)
**File**: `experiments/exp1_baseline.py`

**What to run**:
- Setup: 3 hubs, 5 students per hub (15 total), no drift
- Measure: False alarm rate, communication overhead
- Run 10 times

---

#### ✅ Task 16: Experiment 2 - Sudden Attack (5 hours)
**File**: `experiments/exp2_sudden_attack.py`

**What to run**:
- Setup: Inject blur attack at round 50 to 1 hub (5 students)
- Measure: Detection latency, hub/student identification
- Compare: Hierarchical vs flat federation

---

#### ✅ Task 17: Experiment 3 - Gradual Shift (5 hours)
**File**: `experiments/exp3_gradual_shift.py`

**What to run**:
- Setup: Gradual JPEG compression increase
- Measure: Drift tracking over time

---

#### ✅ Task 18: Experiment 4 - Privacy Trade-off (6 hours)
**File**: `experiments/exp4_privacy_tradeoff.py`

**What to run**:
- Setup: Vary epsilon (0.1, 0.5, 1.0, 5.0, 10.0, ∞)
- Measure: Detection F1 vs privacy budget

---

#### ✅ Task 19: Experiment 5 - Scalability (6 hours)
**File**: `experiments/exp5_scalability.py`

**What to run**:
- Setup: Vary num_hubs (1, 3, 5, 10) and students_per_hub
- Measure: Communication overhead, latency

---

#### ✅ Task 20: Run All Experiments (2 hours)
**File**: `experiments/run_all_experiments.py`

**What to do**:
- Orchestrate all experiments
- Aggregate results
- Generate all plots and tables

---

### **PHASE 4: Paper Writing (Week 6-7)**

#### ✅ Task 21: Write Paper Sections (20 hours total)
- Introduction (4 hours)
- Related Work (6 hours)
- Method (8 hours)
- Experiments (6 hours)
- Conclusion (2 hours)
- Abstract (2 hours)
- Figures & tables (4 hours)

---

### **PHASE 5: Submission (Week 8-9)**

#### ✅ Task 22: Review & Polish (12 hours)
- Internal review
- Revisions
- Formatting
- Submit!

---

## 🔄 Summary: Task Sequence

### Immediate Next Steps (Start Now)
1. ✅ **Task 1**: Privacy Utilities (2-3 hours)
2. ✅ **Task 2**: Sketch Algorithms (4-5 hours)
3. ✅ **Task 3**: Drift Detection (4-5 hours)
4. ✅ **Task 4**: Anomaly Detection (4-5 hours)
5. ✅ **Task 5**: Client Monitor (5-6 hours)

### Week 1 Goal
Complete Tasks 1-5 (core building blocks)

### Week 2 Goal  
Complete Tasks 6-10 (hierarchical federated components)

### Week 3 Goal
Complete Tasks 11-14 (simulation framework)

### Week 4-5 Goal
Complete Tasks 15-20 (all experiments)

### Week 6-7 Goal
Complete Task 21 (paper writing)

### Week 8-9 Goal
Complete Task 22 (review & submit)

---

## 🎓 Research Contribution Enhanced

With the teacher-student hierarchical architecture, our novelty is even stronger:

**Original**: Federated drift detection for deepfakes  
**Enhanced**: **Hierarchical heterogeneous federated drift detection** with teacher-student distillation

**New paper title**: "Privacy-Preserving Hierarchical Drift Detection for Federated Deepfake Forensics"

**Additional contribution**:
- First hierarchical federated learning for deepfake detection
- Knowledge distillation in federated drift monitoring
- Realistic edge deployment (students → hubs → central)

---

## 🚀 Ready to Start!

All documents moved ✅  
Project structure created ✅  
Sequence defined with teacher-student integration ✅  

**Next action**: Implement Task 1 (Privacy Utilities) - shall I begin?
