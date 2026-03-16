# Vehicle-Model-Agnostic Intrusion Detection System (UIDS-II)

A **zero-shot transfer intrusion detection system** for heterogeneous in-vehicle networks (IVNs) using timing-based temporal learning and MobileBERT. Achieves cross-vehicle generalization across ICEV, HEV, BEV platforms and both CAN and CAN-FD protocols **without vehicle-specific retraining**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Latest-green.svg)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

Modern vehicles exhibit extreme heterogeneity across propulsion platforms (ICEV, HEV, BEV), communication protocols (CAN, CAN-FD), and ECU architectures. Conventional intrusion detection systems rely on proprietary CAN identifiers and payload semantics (DBC files), forcing per-vehicle customization and preventing scalable cross-fleet deployment.

This work proposes a **vehicle-model-agnostic IDS** that:
- ✅ Transfers across heterogeneous platforms **without retraining**
- ✅ Uses only timing features (inter-frame and intra-ID deltas)
- ✅ Eliminates DBC and manufacturer-specific dependencies
- ✅ Achieves **F1 = 0.92–0.99** across all vehicle types
- ✅ Maintains **Recall = 1.000** from low injection intensity onward
- ✅ Detects DoS, Fuzzing, Replay, Malfunction, and Spoofing attacks

---

## 📋 Key Technical Contributions

### 1. **Vehicle-Model-Agnostic Feature Mapping**
Instead of raw CAN identifiers and payloads, the system extracts two timing descriptors:
- **Inter-Frame Delta** (Δt_i): Elapsed time between consecutive frames
- **Intra-ID Delta** (Δt^(c)_i): Time since last frame with same identifier

These descriptors are:
- **DBC-independent**: No proprietary signal definitions required
- **Identifier-invariant**: Robust to arbitrary CAN ID relabeling across vehicles
- **Protocol-agnostic**: Work equally on CAN and CAN-FD

### 2. **Ordinal Discretization & Density-Based Windowing**
- Timing descriptors binned into ordinal levels (7 and 9 levels respectively)
- Suppresses noise while preserving temporal structure
- **Adaptive window calibration**: Vehicle-specific temporal window duration computed once at installation
  - Targets 250–280 frames per window
  - Normalizes cross-vehicle traffic density variation

### 3. **MobileBERT-Based Lightweight Classification**
- **Compact Transformer**: 24 layers, 512 hidden dimensions, 8 attention heads
- **4.3× smaller than BERT-BASE**: ~25.3M parameters
- **Bottlenecked architecture**: Reduced per-layer computation for in-vehicle deployment
- **Parameter-efficient fine-tuning**: Only adapter layers and classification head trained

### 4. **Comprehensive Cross-Domain Validation**
- **Cross-vehicle**: ICEV ↔ HEV ↔ BEV transfer without fine-tuning
- **Cross-protocol**: CAN ↔ CAN-FD seamless transfer
- **Cross-laboratory**: LISA → HCRL/DTU zero-adaptation evaluation
- **Cross-attack-type**: Trained on DoS/Fuzz/Replay, generalizes to Malfunction/Spoofing
- **Cross-frequency**: Handles high, medium, low, lower-low, and random injection rates

---

## 🏗️ System Architecture

### End-to-End Pipeline

```
CAN Stream → Timing Extraction → Ordinal Discretization → Token Encoding 
         → Density-Based Windowing → WordPiece Tokenization 
         → MobileBERT Encoder → Binary Classification (Attack/Normal)
```

### Feature Space Transformation

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| **Timing Extraction** | Raw CAN frames (t_i, c_i, d_i) | Δt_i, Δt^(c)_i | Vehicle-independent temporal features |
| **Discretization** | Continuous timing values | Integer bins (0–6, 0–8) | Suppress noise, fixed vocabulary |
| **Token Encoding** | Discretized pairs | String tokens (e.g., "DT3 DG1") | Transformer input format |
| **Windowing** | Frame sequence | Fixed-duration windows | Temporal aggregation |
| **Classification** | Token sequences | Attack probability | Binary decision |

### Vehicle Communication Architectures

```
ICEV (e.g., Kia Soul)
├─ Protocol: CAN (1 Mbps, 8-byte payload)
├─ Key ECUs: ECM, TCM, BCM
└─ CAN IDs: 79 (typical)

HEV (e.g., Chevrolet Silverado)
├─ Protocol: CAN (1 Mbps)
├─ Key ECUs: ECM, HCU, TCM, BCM, BMS
└─ CAN IDs: 98 (typical)

BEV (e.g., Tesla Model 3)
├─ Protocol: CAN (1 Mbps)
├─ Key ECUs: PCM, BMS, Thermal Mgmt
└─ CAN IDs: 69 (typical)

CAN-FD (e.g., Genesis G80)
├─ Protocol: CAN-FD (8 Mbps, 64-byte payload)
├─ Key ECUs: ECM, TCM, BCM, BMS
└─ CAN IDs: 58 (typical)
```

---

## 🔬 Why It Works Across Different Vehicles

### The Problem with Traditional Approaches

Most IDS systems depend on **CAN message IDs and data content**, which vary between vehicle brands:
- Kia Soul uses CAN ID 0x123 for engine data
- Tesla Model 3 uses CAN ID 0x456 for the same purpose
- This difference **forces separate models for each vehicle**

### Our Solution: Focus on Timing, Not Content

Instead, we only look at **when messages arrive**, not what they contain:
- **Inter-Frame Delta**: Time gap between any two consecutive messages
- **Intra-ID Delta**: Time gap between messages with the same ID

These timing patterns:
- ✅ Work the same across all vehicles
- ✅ Don't change regardless of CAN ID values
- ✅ Disrupted in the same way by all injection attacks

### One Model for All Vehicles

- Train on one vehicle (e.g., Kia)
- Use on any other vehicle (Tesla, Genesis, Silverado) **without retraining**
- Only adaptation: Learn the vehicle's natural message rate (10-30 seconds, once)

---

## 📊 Experimental Results

### Cross-Vehicle Evaluation (Zero-Adaptation)

#### Train on Kia (CAN-ICEV) → Test on Others

| Target Vehicle | Protocol | F1-Score | Recall | Precision |
|----------------|----------|----------|--------|-----------|
| Genesis | CAN-FD | **0.974** | 1.000 | 0.949 |
| Tesla | CAN-BEV | **0.880** | 1.000 | 0.786 |
| Silverado | CAN-HEV | **0.943** | 1.000 | 0.892 |

#### Train on Tesla (CAN-BEV) → Test on Others

| Target Vehicle | Protocol | F1-Score | Recall | Precision |
|----------------|----------|----------|--------|-----------|
| Kia | CAN-ICEV | **0.963** | 1.000 | 0.929 |
| Genesis | CAN-FD | **0.985** | 1.000 | 0.971 |
| Silverado | CAN-HEV | **0.976** | 1.000 | 0.954 |

#### Train on Genesis (CAN-FD) → Test on Others

| Target Vehicle | Protocol | F1-Score | Recall | Precision |
|----------------|----------|----------|--------|-----------|
| Kia | CAN-ICEV | **0.977** | 1.000 | 0.956 |
| Tesla | CAN-BEV | **0.922** | 1.000 | 0.855 |
| Silverado | CAN-HEV | **0.944** | 1.000 | 0.894 |

#### Train on Silverado (CAN-HEV) → Test on Others

| Target Vehicle | Protocol | F1-Score | Recall | Precision |
|----------------|----------|----------|--------|-----------|
| Kia | CAN-ICEV | **0.950** | 1.000 | 0.905 |
| Tesla | CAN-BEV | **0.889** | 1.000 | 0.800 |
| Genesis | CAN-FD | **0.999** | 1.000 | 0.998 |

### Cross-Laboratory Validation (LISA → HCRL/DTU)

Training on LISA, evaluating on external HCRL and DTU datasets:

| Train Vehicle | Test Lab | Test Vehicle | F1-Score | Recall |
|--------------|----------|--------------|----------|--------|
| Kia (LISA) | HCRL | Hyundai Sonata | **0.952** | 1.000 |
| Kia (LISA) | DTU | Subaru Forester | **0.918** | 0.998 |
| Tesla (LISA) | HCRL | Hyundai Sonata | **0.945** | 1.000 |
| Tesla (LISA) | DTU | Subaru Forester | **0.931** | 0.999 |

### Zero-Adaptation Unseen Attack Types

Trained on LISA (DoS, Fuzz, Replay) → Tested on HCRL (Malfunction, Spoofing)

| Attack Type | Accuracy | Precision | Recall | F1 |
|------------|----------|-----------|--------|-----|
| Malfunction | 1.00 | 1.00 | 1.00 | 1.00 |
| Gear Spoofing | 0.99 | 1.00 | 0.98 | 0.99 |
| RPM Spoofing | 0.99 | 0.99 | 0.99 | 0.99 |

### Key Performance Properties

✅ **Perfect Recall (1.000)** from Low injection intensity (6.25%) onward across all vehicles  
✅ **F1 = 0.92–0.99** with consistent 95% confidence intervals (CI < 0.08)  
✅ **Attack-Type Agnostic**: DoS, Fuzz, Replay show <0.01 variance at same frequency  
✅ **Frequency-Robust**: Handles low (2.94%) to high (21.05%) and random injection rates  
✅ **Cross-Lab Generalization**: No performance degradation across LISA/HCRL/DTU datasets  

---

## 🛠️ Technical Details (Simplified)

### How Timing Features Work

The system converts continuous time gaps into simple numbers (0-6 or 0-8):

**Inter-Frame Gaps** (time between any messages):
- 0 = very fast (< 50 ms)
- 1 = fast (50-100 ms)
- 2 = medium (100-200 ms)
- ... up to 6 = very slow (> 500 ms)

**Same-ID Gaps** (time since last message with same ID):
- Similar scale from 0-8, but for longer periods
- Captures when each sensor repeats its message

### MobileBERT: The Decision Engine

The model that makes attack/normal decisions:
- **Type**: Compact version of BERT (language model)
- **Size**: ~25.3 MB (4× smaller than standard BERT)
- **Speed**: Analyzes each message window in ~6-7 milliseconds
- **Location**: Runs on vehicle's embedded computer or gateway

### Processing Steps

```
Raw CAN Messages
         ↓
    Extract Timing
         ↓
    Convert to Numbers
         ↓
    Group into Windows
         ↓
   MobileBERT Model
         ↓
  Attack or Normal?
```

---

## 📁 Repository Structure

```
UIDS-II/
│
├── UIDSApp/                              # Streamlit Inference Dashboard
│   ├── OnnxModels/                       # Pre-trained ONNX models (download separately)
│   │   ├── TrainKiaOnnx/                # Kia ICEV models
│   │   ├── TrainGenOnnx/                # Genesis CAN-FD models
│   │   ├── TrainTeslaOnnx/              # Tesla BEV models
│   │   └── TrainSilOnnx/                # Silverado HEV models
│   │
│   ├── app.py                            # Main Streamlit application
│   ├── EvaluateOnnx.py                  # ONNX inference engine
│   ├── utils.py                          # Data preprocessing utilities
│   ├── environment.yml                   # Conda environment
│   └── requirements.txt                  # Pip dependencies
│
├── CANSimulator/                         # Hardware Testbed (C++17)
│   ├── CAN_transmitter.cpp              # Legitimate traffic replay
│   ├── CAN_attacker.cpp                 # Attack injection module
│   ├── Unified_Receiver.cpp             # Real-time monitoring
│   └── CMakeLists.txt                   # Build configuration
│
├── data/                                 # Datasets (not included)
│   ├── LISA/
│   │   ├── kia_soul/
│   │   ├── tesla_model3/
│   │   ├── genesis_g80/
│   │   └── silverado/
│   │
│   ├── HCRL/
│   │   ├── hyundai_sonata/
│   │   └── kia_soul/
│   │
│   └── DTU/
│       ├── subaru_forester/
│       └── silverado/
│
├── docs/
│   ├── images/
│   │   ├── vehicle_architectures.png
│   │   ├── timing_features.png
│   │   └── pipeline.png
│   └── paper.pdf
│
├── environment.yml                       # Conda environment specification
├── requirements.txt                      # Pip dependencies
├── LICENSE                               # MIT License
└── README.md                             # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.10
PyTorch >= 2.0
ONNX Runtime >= 1.14
Streamlit >= 1.0
CUDA >= 11.0 (optional, for GPU acceleration)
```

### Installation

1. **Clone repository**
```bash
git clone https://github.com/Arupreza/UIDS-II.git
cd UIDS-II
```

2. **Download pre-trained models and test data**

📥 **[Download from Google Drive](https://drive.google.com/drive/folders/1GOdTZ0cb4phX_8hPk1nQmYCGQFQj5PjR?usp=sharing)**

Extract and place:
- ONNX models → `UIDSApp/OnnxModels/`
- Datasets → `data/` (optional for evaluation)

3. **Create environment**
```bash
conda env create -f environment.yml
conda activate uidsapp
```

Or with pip:
```bash
pip install -r requirements.txt
```

4. **Launch Streamlit Dashboard**
```bash
cd UIDSApp
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## 💻 How to Use

### Using the Streamlit Dashboard (Recommended)

The easiest way to test the IDS:

1. **Launch the dashboard**
   ```bash
   cd UIDSApp
   streamlit run app.py
   ```

2. **In the web interface**:
   - Select your vehicle model (Kia, Genesis, Tesla, Silverado)
   - Choose a folder containing CAN log files (CSV format)
   - Select CPU or GPU for inference
   - Click "Start" to analyze

3. **View results**:
   - Overall detection metrics (Accuracy, Precision, Recall, F1)
   - Confusion matrix showing correct vs. incorrect detections
   - Per-attack-type performance breakdown

### Input Data Format

Prepare your CAN logs as CSV files with columns:
```
timestamp, can_id, payload
1.234,     0x123,  AB CD EF 01 02 03 04 05
1.245,     0x456,  10 20 30 40 50 60 70 80
```

- **timestamp**: Message arrival time (seconds)
- **can_id**: CAN identifier in hex
- **payload**: Data bytes (space-separated hex)

The system automatically handles the rest.

### What Happens Inside

When you run detection:
1. ✅ Extracts timing features (time between messages)
2. ✅ Converts to ordinal levels (suppresses noise)
3. ✅ Groups into time windows (vehicle-specific duration)
4. ✅ Feeds to MobileBERT model
5. ✅ Outputs attack/normal decision with confidence

**No manual feature engineering needed—fully automated.**

---

## 🧪 Dataset Overview

### Source Datasets

| Dataset | Vehicle | Protocol | Duration | CAN IDs | Attack Types | Frequencies |
|---------|---------|----------|----------|---------|--------------|------------|
| **LISA** | Kia Soul | CAN | 2650 s | 79 | DoS, Fuzz, Replay | High, Med, Low, L-Low |
| **LISA** | Genesis G80 | CAN-FD | 2500 s | 58 | DoS, Fuzz, Replay | High, Med, Low, L-Low |
| **LISA** | Tesla S3 | CAN | 3200 s | 69 | DoS, Fuzz, Replay | High, Med, Low, L-Low |
| **LISA** | Silverado | CAN | 2600 s | 98 | DoS, Fuzz, Replay | High, Med, Low, L-Low |
| **HCRL** | Hyundai Sonata | CAN | 300 k msgs | 64 | DoS, Fuzz, Replay, Mal., Spoof | Multiple |
| **DTU** | Subaru Forester | CAN | 264.5 k msgs | 52 | DoS, Fuzz | Multiple |
| **DTU** | Silverado | CAN | 966 k msgs | 98 | DoS, Fuzz | Multiple |

### Attack Injection Statistics

| Attack Type | Intensity | Total Msgs | Injected | Rate |
|------------|-----------|-----------|----------|------|
| DoS | High | 760,000 | 160,000 | 21.05% |
| | Medium | 700,000 | 100,000 | 14.29% |
| | Low | 640,000 | 40,000 | 6.25% |
| | Lower-Low | 618,100 | 18,179 | 2.94% |
| | Random | 690,362 | 90,362 | 13.09% |
| Fuzz | High–Random | (same pattern as DoS) |
| Replay | High–Random | (same pattern as DoS) |

---

## 📜 What Attacks Does It Detect?

### Supported Attacks

The system is trained to detect:

✅ **Denial-of-Service (DoS)**: Attacker floods the bus with messages  
✅ **Fuzzing**: Random invalid messages injected  
✅ **Replay**: Previously captured messages sent again  
✅ **Malfunction**: Fake data sent to control units  
✅ **Spoofing**: Fake messages pretending to be from other ECUs  

All of these disrupt the normal timing patterns that the system watches for.

### How It Works

The system assumes:
- Attacker has access to CAN bus (through OBD-II port, telematics, etc.)
- Attacker can send any messages they want
- But cannot modify the vehicle's internal computer code

Any injected messages will change **when messages arrive**, which the system detects.

---

## 🏆 Why This System is Different

### Advantages Over Existing Approaches

| Feature | Old Methods | UIDS-II |
|---------|-----------|---------|
| Works on Kia & Tesla | ❌ No | ✅ Yes |
| Needs retraining per car | ✅ Yes | ❌ No |
| Works on CAN & CAN-FD | Partial | ✅ Both |
| Tested on HEV/BEV | ❌ Mostly ICEV | ✅ All types |
| Fast on small devices | Medium | ✅ Very fast |
| Model size | Large (110MB) | ✅ Compact (25MB) |

### Key Strengths

✅ **One model, many vehicles** - Train once, deploy everywhere  
✅ **Works on fast and slow attacks** - No frequency tuning needed  
✅ **No DBC knowledge required** - Works without proprietary data  
✅ **Tested at multiple labs** - Not just academic dataset  
✅ **Detects unseen attack types** - Generalizes beyond training data

---

## 🔧 Setup & Configuration

### First Time Setup

When you use the system on a new vehicle:

1. **Collect 10-30 seconds of normal traffic** (no attacks)
   - This is used to learn the vehicle's natural message rate
   - No manual intervention needed

2. **System automatically calculates window duration**
   - Adapts to vehicle's speed of CAN messages
   - Pre-calculated durations for known vehicles:
     - Kia Soul: 100 ms
     - Genesis: 105 ms  
     - Tesla: 83 ms
     - Silverado: 100 ms

3. **Ready to detect attacks**
   - No additional configuration required
   - Works on any attack frequency

### Automatic Attack Detection

The system handles all attack speeds **without any adjustment**:
- ✅ Very frequent attacks (21% of messages)
- ✅ Frequent attacks (14% of messages)
- ✅ Occasional attacks (6% of messages)
- ✅ Rare attacks (2.94% of messages)
- ✅ Unpredictable attacks (random timing)

Works out-of-the-box with zero tuning.

---

## 📦 Dependencies

### Core Libraries

```
torch>=2.0.0              # Deep learning framework
transformers>=4.30.0      # HuggingFace Transformers (MobileBERT)
onnxruntime>=1.14.0       # ONNX inference optimization
numpy>=1.21.0             # Numerical computing
pandas>=1.3.0             # Data manipulation
streamlit>=1.0.0          # Dashboard UI
scikit-learn>=0.24.0      # Metrics & preprocessing
```

### Optional (for training)

```
pytorch-lightning>=1.6.0   # Training framework
wandb>=0.12.0             # Experiment tracking
pytest>=6.2.0             # Unit testing
```

### CAN Hardware (testbed only)

```
python-can>=3.3.0         # CAN interface library
PEAK PCAN-USB X6          # 6-channel CAN/CAN-FD interface
```

---

## 📈 Performance Metrics Summary

### Overall Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **F1-Score Range** | 0.88–0.99 | Across all vehicle pairs |
| **Recall (Low+)** | 1.000 | From 6.25% injection onward |
| **Precision** | 0.79–1.00 | Varies by vehicle pair |
| **Attack-Type Robustness** | Variance < 0.01 | DoS, Fuzz, Replay uniform |
| **Frequency Robustness** | Variance < 0.05 | Across all injection rates |
| **Cross-Lab Generalization** | F1 = 0.85–1.00 | LISA → HCRL/DTU transfer |
| **Unseen Attack Detection** | F1 = 0.99–1.00 | Malfunction, Spoofing |
| **Model Compression** | 4.3× | vs. BERT-BASE |
| **Inference Latency** | 6.46 ms/window | GPU; 58 ms on Jetson Nano |

---

## 📝 Citation

If you use UIDS-II in your research, please cite:

```bibtex
@article{islam2026vehicle,
  title={Vehicle-Model-Agnostic Intrusion Detection: A Universal Approach for In-Vehicle Networks},
  author={Islam, Md Rezanur and Sarker, Manobendu and Ryu, Donghyun and Yim, Kangbin},
  journal={IEEE Transactions on Vehicular Technology},
  year={2026},
  note={Under Review}
}
```

---

## 🤝 Contributing

Contributions welcome! Please follow standard practices:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -am 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Report Issues

For bugs, feature requests, or questions:
- **GitHub Issues**: [UIDS-II Issues](https://github.com/Arupreza/UIDS-II/issues)
- **Email**: Contact authors below

---

## 👥 Authors

**Md Rezanur Islam**  
*Ph.D. Candidate, Software Convergence*  
Soonchunhyang University, Asan, South Korea  
📧 arupreza@sch.ac.kr

**Manobendu Sarker**  
*Postdoctoral Researcher, Electrical Engineering*  
École de Technologie Supérieure, Montréal, Canada

**Donghyun Ryu**  
*Ph.D. Candidate, Software Convergence*  
Soonchunhyang University, Asan, South Korea  
📧 rdh1999@sch.ac.kr

**Kangbin Yim** (Corresponding Author)  
*Professor, Information Security Engineering*  
Soonchunhyang University, Asan, South Korea  
📧 yim@sch.ac.kr  
🔬 [LISA Lab](https://infolab.soonchunhyang.ac.kr)

---

## 🙏 Acknowledgments

This research was supported by:
- **MSIT** (Ministry of Science and ICT), Korea
- **IITP** (Institute for Information & Communications Technology Planning & Evaluation)
  - Grant: Convergence Security Core Talent Training Business Support Program (IITP-2024-2710008611)
- **Soonchunhyang University** Research Fund

**Datasets**:
- LISA Lab (Soonchunhyang University)
- HCRL (Hacking and Countermeasure Research Lab)
- DTU (Technical University of Denmark)

**Technical Resources**:
- HuggingFace Transformers & MobileBERT
- ONNX Runtime for production inference
- Streamlit for interactive dashboards
- PyTorch & Transformers libraries

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🚗 Disclaimer

This system is designed for **authorized security research and vehicle-fleet IDS deployment only**. Unauthorized modification of vehicle systems or CAN buses may violate laws and void warranties. Always obtain manufacturer and owner consent before testing.

---

**Keywords**: Vehicle-Model Agnostic, Intrusion Detection, CAN Bus Security, Zero-Shot Learning, MobileBERT, Cross-Vehicle Generalization, ICEV, HEV, BEV, CAN-FD, Temporal Learning, Automotive Cybersecurity