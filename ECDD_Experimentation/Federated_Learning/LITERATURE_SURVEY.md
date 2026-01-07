# Literature Survey - Federated Drift Detection for Deepfake Forensics

This document surveys related work and identifies the research gap for our paper.

---

## 1. Deepfake Detection

### Seminal Works
- **FaceForensics++ (2019)**: Benchmark dataset with Face2Face, FaceSwap, Deepfakes, NeuralTextures
- **Capsule Networks (2019)**: Nguyen et al., "Capsule-Forensics: Using Capsule Networks to Detect Forged Images and Videos"
- **XceptionNet (2019)**: Rossler et al., transfer learning from ImageNet for deepfake detection

### Recent Advances
- **LaDeDa (2023)**: Local Attention + Deformable Convolution (basis for our teacher model)
  - Citation: [Need to find exact paper]
  - Key idea: Local patch analysis + attention for generalization
- **Self-supervised approaches (2022-2024)**: Contrastive learning for robust features
- **Frequency analysis (2020)**: Detecting artifacts in frequency domain

### Limitations Addressed by Our Work
- ❌ All assume centralized training/deployment
- ❌ No adaptation to new attack types after deployment
- ❌ Privacy not considered (require raw images for analysis)

---

## 2. Federated Learning

### Foundational Methods
- **FedAvg (2017)**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
  - Simple averaging of client model updates
  - Assumes IID data (unrealistic)

- **FedProx (2020)**: Li et al., "Federated Optimization in Heterogeneous Networks"
  - Adds proximal term to handle heterogeneity
  - Better convergence for non-IID data

- **FedNova (2020)**: Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
  - Normalized averaging (accounts for different local steps)
  - Our existing codebase uses this!

### Privacy-Preserving FL
- **Secure Aggregation (2017)**: Bonawitz et al., cryptographic protocols to protect individual updates
- **Differential Privacy in FL (2019)**: Geyer et al., add DP noise to gradients
  - Trade-off: privacy vs model accuracy
  - We adapt this for monitoring (not training)

### Personalization & Adaptation
- **Personalized FL (2021)**: Multiple approaches (local fine-tuning, meta-learning, clustering)
- **Continual FL (2022)**: Yoon et al., lifelong learning in federated settings
  - Relevant for adapting to new attacks over time

### Gap in FL Literature
- ✅ Training: Well studied
- ❌ **Monitoring/Inference**: Very limited work on federated monitoring of deployed models
- ❌ **Drift detection**: No systematic approach for collaborative drift detection

---

## 3. Drift Detection & Monitoring

### Classical Methods
- **Statistical Tests**:
  - **KS-test (1933)**: Kolmogorov-Smirnov test for distribution comparison
  - **PSI (2000s)**: Population Stability Index, industry standard for model monitoring
  
- **Sequential Change Detection**:
  - **CUSUM (1961)**: Cumulative sum control chart
  - **Page-Hinkley (1954)**: Online change detection algorithm
  - **ADWIN (2007)**: Bifet & Gavaldà, adaptive windowing for concept drift

### Modern Approaches
- **ML-based Drift Detection**:
  - **Autoencoders (2019)**: Reconstruction error for anomaly detection
  - **Uncertainty-based (2020)**: Monitor prediction uncertainty as drift signal
  
- **Data Stream Mining**:
  - **MOA framework (2010)**: Massive Online Analysis for streaming data
  - **River (2020)**: Online machine learning library with drift detection

### Application Domains
- ✅ Traditional ML: Fraud detection, spam filtering, recommendation systems
- ❌ **Media forensics**: Very limited prior work
- ❌ **Federated setting**: No prior work on drift detection in FL

---

## 4. Privacy-Preserving Monitoring

### Differential Privacy
- **Basic DP (2006)**: Dwork et al., foundational definition
- **Local DP (2008)**: Privacy preserved at client before sending to server
- **Laplace Mechanism**: Add Lap(Δf/ε) noise to query result
- **Gaussian Mechanism**: For (ε, δ)-DP, add N(0, σ²) noise

### Sketching Algorithms
- **Count-Min Sketch (2005)**: Cormode & Muthukrishnan, approximate frequency counts
- **HyperLogLog (2007)**: Flajolet et al., cardinality estimation
- **Histograms with DP (2010)**: Hay et al., differentially private histograms
  - Directly applicable to our score distributions

### Privacy-Preserving Aggregation
- **Secure Multi-Party Computation (2008)**: Cryptographic protocols for aggregation
- **Homomorphic Encryption (2011)**: Compute on encrypted data
- **Practical Systems**: Prio (2017), RAPPOR (2014) - deployed at scale by Google, Mozilla

### Our Contribution
- ✅ Apply DP sketching to deepfake score distributions
- ✅ Privacy-preserving drift detection (not training)
- ✅ Trade-off analysis: privacy budget vs detection accuracy

---

## 5. Federated Anomaly Detection

### Related Work (Very Limited!)
- **Federated Outlier Detection (2021)**: Liu et al., "Privacy-Preserving Federated Outlier Detection"
  - Focus: Detect outlier data points
  - Difference: We detect distribution shift, not individual outliers

- **Collaborative Intrusion Detection (2020)**: Zhang et al., federated learning for network security
  - Similar motivation: Multiple organizations collaborate on threat detection
  - Difference: Network traffic vs multimedia forensics

- **Federated Fraud Detection (2022)**: Chen et al., banks collaborate on fraud detection
  - Privacy-preserving aggregation of fraud signals
  - Similar to our approach but different domain

### Research Gap
- ❌ **No prior work on federated drift detection for media forensics**
- ❌ **No system for collaborative deepfake monitoring with privacy guarantees**
- This is our main novelty!

---

## 6. Adaptive Thresholding & Calibration

### Classical Approaches
- **ROC Curve Analysis**: Select threshold based on FPR/TPR trade-off
- **Youden's Index**: Maximize (Sensitivity + Specificity - 1)
- **Cost-sensitive Thresholding**: Incorporate misclassification costs

### Online Calibration
- **Platt Scaling (1999)**: Calibrate classifier outputs to probabilities
- **Isotonic Regression (2005)**: Non-parametric calibration
- **Temperature Scaling (2017)**: Guo et al., simple and effective for neural networks

### Adaptive Systems
- **Online Threshold Optimization (2018)**: Adjust threshold based on streaming data
- **Concept Drift Adaptation (2019)**: Retrain or recalibrate when drift detected

### Our Contribution
- ✅ Federated threshold calibration: Optimize threshold from aggregated distributions
- ✅ Triggered adaptation: Update only when drift detected (stability)

---

## 7. Edge Deployment for Deepfake Detection

### Hardware Platforms
- **Mobile Devices**: MobileNet, EfficientNet architectures
- **Embedded Systems**: Raspberry Pi, Jetson Nano, Arduino (our target)
- **Specialized Hardware**: Google Coral TPU, Intel Neural Compute Stick

### Model Compression
- **Quantization**: INT8, INT4 quantization for reduced memory/computation
- **Pruning**: Remove redundant weights
- **Knowledge Distillation**: Compress large model into smaller student (we use this!)

### Challenges
- Limited compute: Tiny models (<1MB) required for Arduino
- Limited bandwidth: Federated learning amplifies communication bottleneck
- Heterogeneity: Different devices have different capabilities

### Our System
- ✅ Teacher-student architecture: Already optimized for edge
- ✅ Sketch-based monitoring: Minimal communication overhead
- ✅ Heterogeneity-aware: Clients send summaries, not raw data

---

## 8. Deepfake Detection Benchmarks & Datasets

### Major Datasets
- **FaceForensics++ (2019)**: 1.8M frames, 4 manipulation methods
- **CelebDF (2020)**: 5,639 videos, higher quality fakes
- **DFDC (2020)**: Facebook challenge, 124k videos
- **WildDeepfake (2021)**: In-the-wild deepfakes from internet

### Evaluation Protocols
- **Cross-dataset generalization**: Train on one dataset, test on another
- **Cross-manipulation generalization**: Train on Face2Face, test on FaceSwap
- **Perturbation robustness**: Test on compressed, blurred, resized videos

### Our Experiments
- ✅ Use CelebDF (already in our dataset splits)
- ✅ ECDD edge cases simulate new attacks (realistic)
- ✅ Cross-client generalization (federated setting unique challenge)

---

## 9. Related Systems & Tools

### Open-Source Deepfake Detection
- **Deepfake Detection Challenge (DFDC)**: Competition code and models
- **FaceForensics++ Models**: Pre-trained XceptionNet, EfficientNet
- **Sensity.ai**: Commercial deepfake detection API

### Federated Learning Frameworks
- **TensorFlow Federated (TFF)**: Google's FL framework
- **PySyft**: OpenMined's privacy-preserving ML library
- **Flower**: Scalable FL framework (easy to use)
- **FedML**: Research-oriented FL platform

### Our Implementation Choice
- ✅ Custom implementation (full control, simpler for research)
- ✅ REST API for client-server (standard, widely compatible)
- ⏳ Future: Could integrate with Flower for production scalability

---

## 10. Threat Models & Security

### Adversarial Attacks on Deepfake Detectors
- **Adversarial Perturbations (2020)**: Add imperceptible noise to fool detectors
- **GAN-based Attacks (2021)**: Generate adversarially robust deepfakes
- **Model Evasion (2022)**: Reverse-engineer detector to bypass

### Byzantine Attacks in FL
- **Poisoning Attacks (2019)**: Malicious clients send bad updates
- **Backdoor Attacks (2020)**: Inject triggers during training
- **Defense Mechanisms**: Robust aggregation (Krum, Trimmed Mean, Median)

### Our Threat Model
- **Assumption**: Honest-but-curious clients (follow protocol, but may collude)
- **Privacy Guarantee**: Differential privacy (ε=1.0)
- **Future Work**: Byzantine-robust aggregation (mentioned in design, not implemented yet)

---

## Research Gap Analysis

| Research Area | Existing Work | Our Contribution |
|---------------|---------------|------------------|
| Deepfake Detection | ✅ Extensive (LaDeDa, Capsule, etc.) | ✅ Build on existing (use LaDeDa) |
| Federated Learning | ✅ Training well-studied | ✅✅ **Monitoring (novel!)** |
| Drift Detection | ✅ Classical methods exist | ✅✅ **Federated drift detection (novel!)** |
| Privacy-Preserving ML | ✅ DP, secure aggregation | ✅ Apply to drift monitoring |
| Anomaly Detection | ⚠️ Limited FL work | ✅✅ **Clustering-based collaborative detection (novel!)** |
| Adaptive Systems | ✅ Threshold calibration | ✅✅ **Federated threshold adaptation (novel!)** |
| Edge Deployment | ✅ Model compression | ✅ Combine with federated monitoring |
| Media Forensics + FL | ❌ **No prior work!** | ✅✅✅ **Main novelty!** |

**Legend**: ✅ Exists, ⚠️ Limited, ❌ Missing, ✅✅ Our contribution (moderate novelty), ✅✅✅ Our contribution (high novelty)

---

## Key Papers to Cite

### Must-Cite (Core Background)
1. **FedAvg**: McMahan et al., 2017 (foundational FL)
2. **Differential Privacy**: Dwork et al., 2006 (privacy definition)
3. **FaceForensics++**: Rossler et al., 2019 (deepfake benchmark)
4. **LaDeDa**: [Find paper] (our base model)
5. **Drift Detection Survey**: Gama et al., 2014 (comprehensive survey)

### Should-Cite (Related Techniques)
6. **FedNova**: Wang et al., 2020 (our FL method)
7. **Private Histograms**: Hay et al., 2010 (DP sketching)
8. **KS-test**: Massey, 1951 (statistical test)
9. **DBSCAN**: Ester et al., 1996 (clustering algorithm)
10. **Knowledge Distillation**: Hinton et al., 2015 (teacher-student)

### Good-to-Cite (Positioning)
11. **Federated Outlier Detection**: Liu et al., 2021 (related FL work)
12. **Online Learning**: Bottou, 1998 (streaming setting)
13. **Model Monitoring**: Rabanser et al., 2019 (drift detection survey)
14. **Secure Aggregation**: Bonawitz et al., 2017 (privacy mechanism)
15. **CelebDF**: Li et al., 2020 (our dataset)

---

## Related Work Section Outline (for Paper)

### 2.1 Deepfake Detection
- Evolution: Early methods (face warping artifacts) → Deep learning (XceptionNet) → Attention-based (LaDeDa)
- Limitations: Centralized, no adaptation, privacy not considered
- **Transition**: Need for distributed, adaptive, privacy-preserving detection

### 2.2 Federated Learning
- Training: FedAvg, FedProx, FedNova (cite our existing use)
- Privacy: Differential privacy, secure aggregation
- Gap: **Monitoring and inference** in federated settings underexplored

### 2.3 Drift Detection
- Statistical methods: KS-test, PSI, ADWIN
- Applications: Fraud detection, spam filtering, recommendation systems
- Gap: **Not applied to media forensics**, especially not in federated setting

### 2.4 Privacy-Preserving Monitoring
- Sketching: Count-Min Sketch, histograms with DP
- Aggregation: Secure multi-party computation, homomorphic encryption
- Our approach: DP histograms for score distributions (practical trade-off)

### 2.5 Positioning Our Work
- **Unique combination**: Federated + Drift Detection + Privacy + Deepfakes
- **Novel problem**: Collaborative monitoring without data sharing
- **Practical system**: Simulation + real deployment path (Pi + Nicla)

---

## Next: Detailed Implementation Roadmap

Now that design decisions are locked and related work is surveyed, we can create a detailed implementation plan with task breakdown, dependencies, and timeline.

