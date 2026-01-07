# 🚀 Usage Guide - Federated Drift Detection System

**For Engineering Team**  
**Last Updated**: 2026-01-07

---

## 📋 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
# - torch, torchvision (already installed)
# - numpy, scipy, scikit-learn
# - flask, requests
# - matplotlib, seaborn, plotly, dash
# - tqdm, pyyaml
```

---

## 🎯 Step-by-Step Usage

### **STEP 1: Prepare Your Models**

#### Load Your Trained Models
```python
import torch
from deepfake_patch_audit.models.student.tiny_ladeda import TinyLaDeDa
from deepfake_patch_audit.models.teacher.ladeda_wrapper import LaDeDaWrapper

# 1. Load Student Model (Tiny LaDeDa)
student_model = TinyLaDeDa(
    num_classes=1,
    # Add your model-specific parameters here
)
student_model.load_state_dict(
    torch.load('../../deepfake-patch-audit/outputs/checkpoints_two_stage/student_final.pt')
)
student_model.eval()

# 2. Load Teacher Model (LaDeDa)
teacher_model = LaDeDaWrapper(
    # Add your model-specific parameters here
)
teacher_model.load_state_dict(
    torch.load('../../deepfake-patch-audit/outputs/checkpoints_teacher/teacher_finetuned_best.pth')
)
teacher_model.eval()

print("✓ Models loaded successfully")
```

**Model Paths:**
- Student: `Team-Converge/deepfake-patch-audit/outputs/checkpoints_two_stage/student_final.pt`
- Teacher: `Team-Converge/deepfake-patch-audit/outputs/checkpoints_teacher/teacher_finetuned_best.pth`

---

### **STEP 2: Load Your Dataset**

#### Option A: Use CelebDF Test Set
```python
from deepfake_patch_audit.datasets.celebdf_dataset import CelebDFDataset
from torchvision import transforms

# Define transforms (use your existing preprocessing)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = CelebDFDataset(
    root='../../deepfake-patch-audit/data/celebdf',
    split='test',
    transform=transform
)

print(f"✓ Dataset loaded: {len(dataset)} samples")
```

#### Option B: Use Custom Dataset
```python
from torch.utils.data import Dataset
import os
from PIL import Image

class CustomDeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load file paths and labels
        self.samples = []
        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(root_dir, folder)
            for img_name in os.listdir(folder_path):
                self.samples.append((
                    os.path.join(folder_path, img_name),
                    label
                ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Use it
dataset = CustomDeepfakeDataset(
    root_dir='path/to/your/test/data',
    transform=transform
)
```

---

### **STEP 3: Compute Baseline Distribution**

This is **CRITICAL** for drift detection!

```python
import numpy as np
from torch.utils.data import DataLoader

def compute_baseline_histogram(model, val_loader, device='cpu'):
    """
    Compute baseline score distribution from validation set.
    
    Args:
        model: Your teacher model
        val_loader: Validation data loader
        device: 'cpu' or 'cuda'
        
    Returns:
        baseline_hist: Normalized histogram (20 bins)
    """
    model.eval()
    model.to(device)
    
    all_scores = []
    
    print("Computing baseline distribution...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get scores (sigmoid for binary classification)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores.flatten().tolist())
    
    # Create histogram (20 bins, range [0, 1])
    baseline_hist, bin_edges = np.histogram(all_scores, bins=20, range=(0, 1))
    baseline_hist = baseline_hist / baseline_hist.sum()  # Normalize
    
    print(f"✓ Baseline computed: mean={np.mean(all_scores):.3f}, std={np.std(all_scores):.3f}")
    
    return baseline_hist

# Create validation loader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Compute baseline
baseline_hist = compute_baseline_histogram(teacher_model, val_loader)

# Save baseline for future use
np.save('baseline_histogram.npy', baseline_hist)
```

**Important Notes:**
- Use **validation set** (not test set) for baseline
- Use **teacher model** for baseline computation
- Save baseline - you'll need it for all experiments!

---

### **STEP 4: Run Experiments**

#### Option A: Run All Experiments (Recommended)
```bash
cd experiments

# Quick test (2-3 hours)
python run_all_experiments.py \
    --student_model ../../deepfake-patch-audit/outputs/checkpoints_two_stage/student_final.pt \
    --teacher_model ../../deepfake-patch-audit/outputs/checkpoints_teacher/teacher_finetuned_best.pth \
    --dataset ../../deepfake-patch-audit/data/celebdf/test \
    --quick

# Full experiments (20-30 hours)
python run_all_experiments.py \
    --student_model ../../deepfake-patch-audit/outputs/checkpoints_two_stage/student_final.pt \
    --teacher_model ../../deepfake-patch-audit/outputs/checkpoints_teacher/teacher_finetuned_best.pth \
    --dataset ../../deepfake-patch-audit/data/celebdf/test \
    --save_dir ../results
```

#### Option B: Run Individual Experiments
```python
# experiments/run_single_experiment.py
from exp1_baseline import run_baseline_experiment
from exp2_sudden_attack import run_sudden_attack_experiment
# ... import others

# Load your models and data (see STEP 1-3)
student_model = load_your_student_model()
teacher_model = load_your_teacher_model()
dataset = load_your_dataset()
baseline_hist = load_baseline_histogram()

# Run specific experiment
results = run_baseline_experiment(
    student_model=student_model,
    teacher_model=teacher_model,
    dataset=dataset,
    baseline_hist=baseline_hist,
    num_runs=10,
    save_dir='results/exp1_baseline'
)

print(f"Experiment complete: {results}")
```

---

### **STEP 5: Monitor Progress**

#### Real-Time Dashboard
```bash
cd visualization

# Start dashboard
python dashboard.py

# Open browser to: http://localhost:8050/
```

#### Load Results into Dashboard
```python
from visualization.dashboard import FederatedMonitoringDashboard

dashboard = FederatedMonitoringDashboard(port=8050)

# Load from results file
dashboard.load_from_results('../results/exp1_baseline/run_0.json')

# Run
dashboard.run()
```

---

### **STEP 6: Generate Visualizations**

```bash
cd visualization

# Generate all plots and HTML dashboard
python static_visualizer.py ../results/

# This creates:
# - results/figures/*.png (all plots)
# - results/index.html (interactive dashboard)
# - results/summary_table.csv
```

---

## 🔧 Advanced Usage

### Custom Experiment Configuration

```python
# experiments/custom_experiment.py
from simulation.fed_drift_simulator import HierarchicalFedSimulator
from simulation.drift_scenarios import SuddenAttackScenario
from simulation.evaluation_metrics import ExperimentMetrics

# Configure your experiment
config = {
    'num_hubs': 3,
    'students_per_hub': 5,
    'num_rounds': 100,
    'dropout_rate': 0.2,
    'non_iid': True,
    'powerlaw': True
}

# Create simulator
simulator = HierarchicalFedSimulator(
    student_model=your_student_model,
    teacher_model=your_teacher_model,
    dataset=your_dataset,
    baseline_hist=your_baseline_hist,
    num_hubs=config['num_hubs'],
    students_per_hub=config['students_per_hub'],
    non_iid=config['non_iid'],
    powerlaw=config['powerlaw'],
    dropout_rate=config['dropout_rate']
)

# Create drift scenario
scenario = SuddenAttackScenario(
    attack_type='blur',      # 'blur', 'jpeg', or 'resize'
    intensity=0.3,           # 30% of samples affected
    start_round=50,          # Inject at round 50
    affected_clients=[0, 1, 2]  # Which clients to affect
)

# Run experiment
results = simulator.run_experiment(
    num_rounds=config['num_rounds'],
    scenario=scenario,
    predictions_per_round=10,
    verbose=True
)

# Save results
simulator.save_results('custom_experiment_results.json')
```

---

## 📊 Understanding Results

### Result Files Structure
```
results/
├── experiments_20260107_143000/
│   ├── exp1_baseline/
│   │   ├── run_0.json          # Individual run
│   │   ├── run_1.json
│   │   ├── ...
│   │   └── aggregated_results.json  # Summary
│   ├── exp2_sudden_attack/
│   ├── exp3_gradual_shift/
│   ├── exp4_privacy_tradeoff/
│   ├── exp5_scalability/
│   ├── exp6_privacy_audit/
│   └── master_results.json     # All experiments summary
```

### Reading Results
```python
import json

# Load aggregated results
with open('results/exp1_baseline/aggregated_results.json', 'r') as f:
    results = json.load(f)

# Access metrics
print(f"False alarm rate: {results['false_alarm_rate_mean']:.3f}")
print(f"Communication overhead: {results['total_bytes_mean']:.0f} bytes")

# Load master results
with open('results/master_results.json', 'r') as f:
    master = json.load(f)

# Compare experiments
for exp_name, exp_results in master['experiments'].items():
    print(f"{exp_name}: {exp_results}")
```

---

## 🎯 Common Use Cases

### Use Case 1: Quick Validation
**Goal**: Verify system works with your models

```bash
# 1. Quick test with 2 runs per experiment
python experiments/run_all_experiments.py --quick

# 2. Check results
ls results/experiments_*/

# 3. Generate visualizations
python visualization/static_visualizer.py results/
```

---

### Use Case 2: Full Evaluation for Paper
**Goal**: Complete experiments for publication

```bash
# 1. Run all experiments (overnight job)
nohup python experiments/run_all_experiments.py \
    --student_model <path> \
    --teacher_model <path> \
    --dataset <path> \
    > experiments.log 2>&1 &

# 2. Monitor progress
tail -f experiments.log

# 3. When complete, generate all figures
python visualization/static_visualizer.py results/

# 4. Results ready for paper!
```

---

### Use Case 3: Privacy Analysis Only
**Goal**: Analyze privacy guarantees

```python
# experiments/privacy_only.py
from experiments.exp6_privacy_audit import run_privacy_audit

results = run_privacy_audit(
    student_model=your_student,
    teacher_model=your_teacher,
    dataset=your_dataset,
    baseline_hist=your_baseline,
    epsilon_values=[0.1, 0.5, 1.0, 5.0, 10.0],
    num_runs_per_epsilon=5,
    save_dir='results/privacy_audit'
)

# Check privacy guarantees
for epsilon, audit in results.items():
    print(f"ε={epsilon}:")
    print(f"  Reconstruction error: {audit['aggregated']['mean_reconstruction_error']:.4f}")
    print(f"  Membership advantage: {audit['aggregated']['mean_membership_advantage']:.4f}")
```

---

### Use Case 4: Test Single Drift Scenario
**Goal**: Test specific attack type

```python
# test_blur_attack.py
from simulation.drift_scenarios import SuddenAttackScenario

# Create blur attack
scenario = SuddenAttackScenario(
    attack_type='blur',
    intensity=0.5,  # 50% of samples
    start_round=30,
    affected_clients=[0, 1, 2, 3, 4]  # 5 clients affected
)

# Run with your simulator
results = simulator.run_experiment(
    num_rounds=100,
    scenario=scenario
)

# Check if detected
print(f"Drift detected: {results['drift_detected']}")
print(f"Detection latency: {results['central_server_report']['drift_scores']}")
```

---

## 🐛 Troubleshooting

### Problem 1: Model Loading Errors
```python
# If you get "missing keys" error
student_state = torch.load('student_final.pt')
print(student_state.keys())  # Check what keys are in checkpoint

# May need to extract model weights
if 'model_state_dict' in student_state:
    student_model.load_state_dict(student_state['model_state_dict'])
else:
    student_model.load_state_dict(student_state)
```

### Problem 2: Dataset Format Issues
```python
# Check dataset format
dataset = your_dataset
print(f"Dataset length: {len(dataset)}")
print(f"Sample format: {type(dataset[0])}")

# Should return (image_tensor, label) or (image, label)
image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label}")
```

### Problem 3: Out of Memory
```python
# Reduce batch size in simulator
simulator = HierarchicalFedSimulator(
    ...,
    # Reduce these if OOM
    students_per_hub=3,  # Instead of 5
    num_hubs=2,          # Instead of 3
)

# Or run experiments on CPU
device = 'cpu'  # Instead of 'cuda'
```

### Problem 4: Slow Experiments
```python
# Speed up for testing
config = {
    'num_rounds': 20,            # Instead of 100
    'predictions_per_round': 5,  # Instead of 10
    'num_runs': 2                # Instead of 10
}
```

---

## 📁 File Locations Reference

### Your Existing Models & Data
```
Team-Converge/deepfake-patch-audit/
├── outputs/
│   ├── checkpoints_teacher/
│   │   └── teacher_finetuned_best.pth       ← Teacher model
│   └── checkpoints_two_stage/
│       ├── student_final.pt                  ← Student model
│       └── calibration/
│           └── threshold_calibration.json    ← Use for baseline threshold
├── models/
│   ├── student/
│   │   └── tiny_ladeda.py                    ← Student architecture
│   └── teacher/
│       └── ladeda_wrapper.py                 ← Teacher architecture
└── datasets/
    └── celebdf_dataset.py                    ← Dataset loader
```

### Federated Learning System
```
Team-Converge/ECDD_Experimentation/Federated_Learning/
├── core/                    ← Privacy, drift detection, anomaly
├── client/                  ← Student client, monitoring
├── server/                  ← Hub aggregator, central server
├── simulation/              ← Simulator, scenarios, metrics
├── experiments/             ← 6 experiments + master script
├── privacy/                 ← Privacy analysis
├── visualization/           ← Dashboards
└── results/                 ← Experiment outputs (created when run)
```

---

## ✅ Checklist Before Running

- [ ] Models loaded successfully (student + teacher)
- [ ] Dataset loaded and verified (check `len(dataset)`)
- [ ] Baseline histogram computed and saved
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space for results (~1-2 GB)
- [ ] Time allocated (2-3 hours for quick, 20-30 hours for full)

---

## 🆘 Getting Help

### Check Documentation
1. `README.md` - Overview
2. `DESIGN_DECISIONS.md` - All design choices
3. `IMPLEMENTATION_ROADMAP.md` - Detailed plan
4. `FINAL_STATUS.md` - Complete summary

### Common Issues
- **Import errors**: Make sure you're in the right directory
- **Model incompatibility**: Check model architectures match
- **Dataset errors**: Verify dataset path and format
- **Slow performance**: Use `--quick` flag for testing

### Debug Mode
```python
# Enable verbose output
results = simulator.run_experiment(
    num_rounds=10,
    verbose=True  # Print detailed logs
)

# Check intermediate results
print(f"Rounds completed: {len(simulator.round_logs)}")
print(f"Last round: {simulator.round_logs[-1]}")
```

---

## 📝 Example Complete Workflow

```python
#!/usr/bin/env python3
"""
complete_workflow.py - Full example from start to finish
"""

import torch
import numpy as np
from pathlib import Path

# STEP 1: Load models
print("Step 1: Loading models...")
from deepfake_patch_audit.models.student.tiny_ladeda import TinyLaDeDa
from deepfake_patch_audit.models.teacher.ladeda_wrapper import LaDeDaWrapper

student = TinyLaDeDa()
student.load_state_dict(torch.load('path/to/student_final.pt'))
student.eval()

teacher = LaDeDaWrapper()
teacher.load_state_dict(torch.load('path/to/teacher_finetuned_best.pth'))
teacher.eval()

# STEP 2: Load dataset
print("Step 2: Loading dataset...")
from deepfake_patch_audit.datasets.celebdf_dataset import CelebDFDataset

dataset = CelebDFDataset(root='path/to/celebdf', split='test')

# STEP 3: Compute baseline
print("Step 3: Computing baseline...")
baseline_hist = np.load('baseline_histogram.npy')  # Or compute it

# STEP 4: Run experiments
print("Step 4: Running experiments...")
from experiments.run_all_experiments import run_all_experiments

results = run_all_experiments(
    student_model=student,
    teacher_model=teacher,
    dataset=dataset,
    baseline_hist=baseline_hist,
    save_dir='results',
    quick_mode=True  # Set False for full experiments
)

print("✓ Complete! Check results/ directory")

# STEP 5: Generate visualizations
print("Step 5: Generating visualizations...")
import subprocess
subprocess.run(['python', 'visualization/static_visualizer.py', 'results/'])

print("✓ Done! Open results/index.html to view results")
```

---

## 🎉 You're Ready!

Follow the steps above to run the complete federated drift detection system with your trained models!

**Questions?** Check the documentation files or review the code comments.

**Good luck with your experiments!** 🚀
