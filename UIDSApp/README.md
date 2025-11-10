# 🧠 UIDSApp: Universal Intrusion Detection System for In-Vehicle Network

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A **Streamlit-based dashboard** for evaluating and running inference on multiple ONNX models trained for vehicle intrusion detection across different vehicle types (Kia, Genesis, Tesla, Silverado). The app enables researchers and engineers to interactively select models, configure parameters, and evaluate CAN bus datasets.

---

## 🎯 Project Overview

This repository implements a production-ready evaluation and inference platform for ONNX-exported deep learning models. The project demonstrates modern ML deployment practices with interactive dashboards and comprehensive model evaluation capabilities for automotive cybersecurity applications.

## ✨ Key Features

- **Multi-Model Support**: Evaluate ONNX models for Kia, Genesis, Tesla, and Silverado vehicles
- **Interactive Dashboard**: Streamlit-powered web interface for real-time model evaluation
- **Flexible Inference Modes**: 
  - Validation mode (with ground truth labels)
  - Real-life inference mode (production deployment)
- **Hardware Optimization**: Configurable CPU or CUDA GPU acceleration via ONNX Runtime
- **Comprehensive Metrics**: 
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix visualization
  - Per-class performance analysis
- **Time-Series Processing**: Configurable time gap parameter for CAN message segmentation
- **Production Ready**: Environment-managed and deployment-friendly

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Arupreza/UIDS-II.git
cd UIDS-II/UIDSApp
```

### 2️⃣ Download ONNX Models and Test Data

Download the pre-trained ONNX models and test datasets from Google Drive:

**[📥 Download Models & Test Data](https://drive.google.com/drive/folders/1GOdTZ0cb4phX_8hPk1nQmYCGQFQj5PjR?usp=sharing)**

Extract the contents and place them in the appropriate directories:
- ONNX models → `UIDSApp/OnnxModels/`
- Test data → Your preferred location (specify path in the dashboard)

### 3️⃣ Create Environment from YAML

All dependencies are managed using `environment.yml`:

```bash
conda env create -f environment.yml
```

### 3️⃣ Activate Environment

```bash
conda activate uidsapp
```

### 4️⃣ Launch the Streamlit App

```bash
streamlit run app.py
```

The web dashboard will automatically open in your browser (default: `http://localhost:8501`).

---

## ⚙️ Environment Details

The environment configuration installs all necessary dependencies:

- **Python 3.10** - Core runtime
- **Streamlit** - Web interface framework
- **ONNX Runtime** - Optimized model inference (CPU/CUDA)
- **NumPy / Pandas** - Data processing and manipulation
- **tqdm** - Progress bar visualization
- **Scikit-learn** - Evaluation metrics and utilities
- **Transformers** - Tokenizer and preprocessing (if needed)

**Environment Configuration** (`environment.yml`):

```yaml
name: uidsapp
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - streamlit
      - onnxruntime
      - numpy
      - pandas
      - tqdm
      - scikit-learn
      - transformers
```

**Alternative Installation** (using pip):

```bash
pip install -r requirements.txt
```

---

## 📂 Directory Structure

```
UIDSApp/
│
├── OnnxModels/              # Pre-trained ONNX models (download from Drive)
│   ├── TrainKiaOnnx/       # Kia vehicle models
│   ├── TrainGenOnnx/       # Genesis vehicle models
│   ├── TrainTeslaOnnx/     # Tesla vehicle models
│   └── TrainSilOnnx/       # Silverado vehicle models
│
├── app.py                   # Streamlit main dashboard
├── EvaluateOnnx.py         # ONNX model loading & evaluation logic
├── utils.py                 # Helper utilities and data processing
├── environment.yml          # Conda environment specification
└── requirements.txt         # Alternative pip dependencies
```

> **Note**: ONNX models and test data must be downloaded separately from [Google Drive](https://drive.google.com/drive/folders/1GOdTZ0cb4phX_8hPk1nQmYCGQFQj5PjR?usp=sharing)

---

## 🧠 How to Use

### 1️⃣ Select Vehicle Model

Choose your target vehicle type:
- **Kia**
- **Genesis**
- **Tesla**
- **Silverado**

The app automatically loads the corresponding ONNX model from:
```
UIDSApp/OnnxModels/Train<ModelName>Onnx/
```

### 2️⃣ Configure Run Parameters

| Parameter | Description |
|-----------|-------------|
| **Path to CAN CSV Folder** | Directory containing CAN-bus `.csv` files for evaluation |
| **Time Gap (seconds)** | Controls message segmentation window for time-series analysis |
| **ONNX Runtime Device** | Hardware acceleration: `cpu` or `cuda` (GPU) |
| **Select Mode** | `Validation` (with labels) or `Real-Life Inference` (no labels) |

### 3️⃣ Run Evaluation or Inference

Click **Start** to begin processing:

1. **Load** the selected ONNX model
2. **Read** all CAN CSV files in the specified folder
3. **Perform** evaluation or inference based on selected mode
4. **Display** comprehensive results:
   - Overall accuracy and metrics
   - Per-class precision, recall, F1-score
   - Confusion matrix visualization
   - Prediction distribution analysis

---

## 📊 Model Evaluation Workflow

1. **Data Loading**: Reads CAN bus CSV files containing timestamped messages
2. **Preprocessing**: Segments messages based on configurable time gaps
3. **Feature Extraction**: Converts CAN data into model-compatible tensors
4. **Inference**: Runs ONNX Runtime prediction (CPU or GPU accelerated)
5. **Metrics Calculation**: Computes comprehensive evaluation metrics
6. **Visualization**: Displays interactive results in Streamlit dashboard

---

## 🔧 Advanced Configuration

### Download Resources

**ONNX Models and Test Data**: All pre-trained models and evaluation datasets are available on Google Drive:

🔗 **[Download from Google Drive](https://drive.google.com/drive/folders/1GOdTZ0cb4phX_8hPk1nQmYCGQFQj5PjR?usp=sharing)**

The download includes:
- Pre-trained ONNX models for all vehicle types (Kia, Genesis, Tesla, Silverado)
- CAN bus test datasets for evaluation
- Sample data for real-life inference testing

### Custom Model Integration

To add a new vehicle model:

1. Export your trained model to ONNX format
2. Create a new directory: `OnnxModels/Train<VehicleName>Onnx/`
3. Place your `.onnx` file in the directory
4. Update the model selection dropdown in `app.py`

### GPU Acceleration

Ensure CUDA-compatible ONNX Runtime is installed:

```bash
pip install onnxruntime-gpu
```

Select `cuda` in the dashboard's device configuration.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [ONNX Runtime](https://onnxruntime.ai/)
- Designed for automotive cybersecurity research
