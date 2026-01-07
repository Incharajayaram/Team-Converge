# Federated Privacy-Preserving Drift Detection for Deepfake Forensics

This folder contains the implementation of a federated learning system for collaborative deepfake detection with privacy-preserving drift monitoring.

## Key Innovation: Teacher-Student Federated Architecture

Unlike traditional federated learning (homogeneous models), our system leverages the existing **teacher-student distillation paradigm**:

- **Hub devices (Raspberry Pi)**: Run teacher model, act as federated aggregators
- **Edge devices (Arduino Nicla)**: Run student model, act as lightweight clients
- **Hierarchical federation**: Students learn from local data, teachers aggregate and redistribute knowledge

## Project Structure

```
Federated_Learning/
├── README.md (this file)
├── DESIGN_DECISIONS.md (all 10 design choices)
├── LITERATURE_SURVEY.md (related work + research gap)
├── IMPLEMENTATION_ROADMAP.md (9-week plan)
├── FEDERATED_DRIFT_DETECTION_PLAN.md (architecture overview)
│
├── core/ (Core federated components)
│   ├── __init__.py
│   ├── privacy_utils.py (DP mechanisms)
│   ├── sketch_algorithms.py (histograms, statistics)
│   ├── drift_detection.py (KS-test, PSI, JS-divergence)
│   └── anomaly_detection.py (DBSCAN clustering)
│
├── client/ (Client-side components)
│   ├── __init__.py
│   ├── client_monitor.py (local monitoring)
│   ├── federated_client.py (communication wrapper)
│   └── student_client.py (student-specific client)
│
├── server/ (Server-side components)
│   ├── __init__.py
│   ├── drift_server.py (aggregation + drift detection)
│   ├── adaptive_threshold.py (threshold calibration)
│   ├── server_api.py (REST API)
│   └── teacher_aggregator.py (teacher-student aggregation)
│
├── simulation/ (Simulation framework)
│   ├── __init__.py
│   ├── data_partitioning.py (non-IID splits)
│   ├── drift_scenarios.py (attack injection)
│   ├── fed_drift_simulator.py (orchestrator)
│   └── evaluation_metrics.py (metrics + plotting)
│
├── experiments/ (Experiment scripts)
│   ├── __init__.py
│   ├── exp1_baseline.py
│   ├── exp2_sudden_attack.py
│   ├── exp3_gradual_shift.py
│   ├── exp4_privacy_tradeoff.py
│   ├── exp5_scalability.py
│   └── run_all_experiments.py
│
├── tests/ (Unit tests)
│   ├── __init__.py
│   ├── test_privacy_utils.py
│   ├── test_sketch_algorithms.py
│   ├── test_drift_detection.py
│   └── test_simulation.py
│
├── results/ (Experimental results)
│   ├── figures/
│   ├── tables/
│   └── logs/
│
└── configs/ (Configuration files)
    ├── default_config.yaml
    ├── privacy_configs.yaml
    └── experiment_configs.yaml
```

## Teacher-Student Federated Learning

### Standard FL (Baseline)
```
Client 1 (Model) ──┐
Client 2 (Model) ──┼──> Server (Aggregate) ──> Global Model
Client 3 (Model) ──┘
```

### Our Hierarchical FL (Novel)
```
Edge Devices (Students)          Hub Devices (Teachers)         Central Server
─────────────────────            ───────────────────            ──────────────
Nicla 1 (Student) ──┐            
Nicla 2 (Student) ──┼──> Pi 1 (Teacher Agg) ──┐
Nicla 3 (Student) ──┘                           │
                                                 ├──> Central (Global Agg)
Nicla 4 (Student) ──┐                           │
Nicla 5 (Student) ──┼──> Pi 2 (Teacher Agg) ──┘
Nicla 6 (Student) ──┘
```

### Benefits
1. **Reduced communication**: Students send to nearby hub, not central server
2. **Heterogeneous models**: Different architectures at different levels
3. **Knowledge distillation**: Teachers provide soft labels for student learning
4. **Realistic deployment**: Matches actual edge infrastructure (hubs + devices)

## Quick Start

### Installation
```bash
cd Team-Converge/ECDD_Experimentation/Federated_Learning
pip install -r requirements.txt
```

### Run Baseline Experiment
```bash
python experiments/exp1_baseline.py --config configs/default_config.yaml
```

### Run All Experiments
```bash
python experiments/run_all_experiments.py
```

## Documentation

- **Design Decisions**: See `DESIGN_DECISIONS.md`
- **Literature Survey**: See `LITERATURE_SURVEY.md`
- **Implementation Roadmap**: See `IMPLEMENTATION_ROADMAP.md`
- **Architecture Overview**: See `FEDERATED_DRIFT_DETECTION_PLAN.md`

## Research Paper

Target venues: ACM Multimedia 2026, WACV 2027

**Title**: "Privacy-Preserving Hierarchical Drift Detection for Federated Deepfake Forensics"

**Key Contributions**:
1. First federated drift detection system for deepfake forensics
2. Hierarchical teacher-student federated architecture
3. Privacy-preserving sketch-based monitoring with DP guarantees
4. Comprehensive experimental validation with realistic drift scenarios

## License

[To be determined]
