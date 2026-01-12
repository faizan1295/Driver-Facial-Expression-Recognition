# 🚗 DFER-GCViT: Driver Facial Expression Recognition

> Efficient Fine-Tuning for Real-Time Automotive Applications using Global Context Vision Transformer

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research-green.svg)]()

**Author:** Syed Faizan Abbas Masood  
**Institution:** BTU Cottbus-Senftenberg  
**Department:** Graphical Systems

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Project Structure](#-project-structure)
- [Running the Project](#-running-the-project)
- [Model Architecture](#-model-architecture)
- [Training Configuration](#-training-configuration)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This project implements a Driver Facial Expression Recognition system using the DFER-GCViT (Global Context Vision Transformer) architecture. The research focuses on full fine-tuning classification performance for real-time automotive safety applications.

**Key Research Focus:**
- Full fine-tuning strategy using pre-trained GCViT model
- Efficient training on Apple Silicon (M1/M2/M3) using Metal Performance Shaders (MPS)
- Real-time driver emotion detection for automotive safety systems
- KMU-FED dataset for driver expression classification

**Expression Classes:**
- Anger
- Disgust
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

---

## ✨ Features

- ✅ **Apple Silicon Optimized**: Native support for M1/M2/M3 chips using MPS
- ✅ **Pre-trained Backbone**: Leverages GCViT pre-trained weights from ImageNet
- ✅ **Advanced Augmentation**: Albumentations for robust data preprocessing
- ✅ **Training Monitoring**: Real-time progress tracking with tqdm
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- ✅ **Visualization Suite**: Training curves, confusion matrices, class distribution
- ✅ **Mixed Precision Training**: FP16 support for faster training
- ✅ **Learning Rate Scheduling**: Cosine annealing with warm restarts

---

## 📦 Requirements

### Hardware Requirements

**Minimum:**
- CPU: Intel Core i5 or Apple M1
- RAM: 8GB
- Storage: 5GB free space

**Recommended:**
- CPU: Apple M3 Pro or equivalent
- RAM: 16GB+
- GPU: Apple Silicon (MPS) or NVIDIA GPU with CUDA support
- Storage: 10GB free space

### Software Requirements

- **Operating System**: macOS 12+, Linux, or Windows 10+
- **Python**: 3.9 or higher
- **Jupyter Notebook**: Latest version

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
# Create project directory
mkdir dfer-gcvit-project
cd dfer-gcvit-project

# Download the notebook file
# Place DFER_GCViT_Mac.ipynb in this directory
```

### Step 2: Set Up Python Virtual Environment

#### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### Step 3: Install Dependencies

The notebook will automatically install all required packages in the first cell. However, you can manually install them:

```bash
pip install --upgrade pip

pip install numpy torch torchvision timm albumentations h5py opencv-python-headless tqdm pandas matplotlib seaborn scikit-learn jupyter
```

**Core Dependencies:**
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `timm>=0.9.0` - PyTorch Image Models (pre-trained GCViT)
- `albumentations>=1.3.0` - Advanced image augmentation
- `h5py>=3.8.0` - HDF5 file handling
- `opencv-python-headless>=4.7.0` - Image processing
- `tqdm>=4.65.0` - Progress bars
- `pandas>=1.5.0` - Data manipulation
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `scikit-learn>=1.2.0` - Machine learning metrics

---

## 🏗️ Model Architecture

```
Input Image (224×224×3)
         │
         ▼
┌─────────────────────────────────────┐
│     GCViT Backbone (Pretrained)     │
│  ┌─────────────────────────────┐    │
│  │ Stage 1: Local+Global Attn  │    │
│  │ Stage 2: Local+Global Attn  │    │
│  │ Stage 3: Local+Global Attn  │    │
│  │ Stage 4: Local+Global Attn  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Custom Classifier Head          │
│  ┌─────────────────────────────┐    │
│  │ Linear(512 → 256)           │    │
│  │ BatchNorm + ReLU + Dropout  │    │
│  │ Linear(256 → 7)             │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         │
         ▼
    7 Emotion Classes
```

## 📊 Dataset Setup

### Step 1: Download KMU-FED Dataset

The project uses the **KMU-FED (Kookmin University - Facial Expression in Driving)** dataset.

**Option A: Using Kaggle**
```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials (place kaggle.json in ~/.kaggle/)
# Download dataset
kaggle datasets download -d your-dataset-path
```

**Option B: Manual Download**
- Download the dataset from the official source
- The dataset should be in `.zip` format named `archive.zip`

### Step 2: Place Dataset File

```bash
# Create data directory
mkdir -p ~/Downloads/data-exploration

# Place archive.zip in this directory
# ~/Downloads/data-exploration/archive.zip
```

**Expected Dataset Structure (after extraction):**
```
archive/
├── KMU-FED/
│   ├── train/
│   │   ├── anger/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happiness/
│   │   ├── neutral/
│   │   ├── sadness/
│   │   └── surprise/
│   └── validation/
│       ├── anger/
│       ├── disgust/
│       ├── fear/
│       ├── happiness/
│       ├── neutral/
│       ├── sadness/
│       └── surprise/
```

### Step 3: Update Working Directory (Important!)

In the notebook, update the `WORKING_DIR` variable in **Section 2** to match your setup:

```python
# Change this line to your actual path
WORKING_DIR = '/path/to/your/data-exploration'  # Update this!
os.chdir(WORKING_DIR)
```

---

## 📁 Project Structure

```
dfer-gcvit-project/
├── DFER_GCViT_Mac.ipynb          # Main notebook
├── README.md                      # This file
├── venv/                          # Virtual environment
└── data-exploration/              # Working directory
    ├── archive.zip                # Original dataset
    ├── archive/                   # Extracted dataset
    │   └── KMU-FED/
    ├── train_annotations.csv      # Training metadata
    ├── val_annotations.csv        # Validation metadata
    ├── train_data.h5              # Preprocessed training data
    ├── val_data.h5                # Preprocessed validation data
    ├── best_model.pth             # Best trained model
    ├── confusion_matrix.png       # Confusion matrix visualization
    ├── performance_summary.png    # Performance charts
    └── final_results.json         # Training results
```

---

## 🏃 Running the Project

### Option 1: Using Jupyter Notebook (Recommended)

1. **Launch Jupyter Notebook:**
   ```bash
   # Make sure virtual environment is activated
   jupyter notebook
   ```

2. **Open the notebook:**
   - Navigate to `DFER_GCViT_Mac.ipynb`
   - Click to open

3. **Execute cells sequentially:**
   - **Cell 1-3**: Setup & Installation - Installs dependencies and checks GPU availability
   - **Cell 4**: Data Upload - Checks for dataset file
   - **Cell 5**: Data Extraction - Extracts dataset from zip
   - **Cell 6**: Data Exploration - Analyzes class distribution
   - **Cell 7**: Data Preprocessing - Creates HDF5 files for efficient loading
   - **Cell 8-9**: Model Architecture - Defines GCViT model with custom classification head
   - **Cell 10**: Training Configuration - Sets hyperparameters and strategy
   - **Cell 11**: Model Training - Trains for 50 epochs with full fine-tuning
   - **Cell 12**: Model Evaluation - Tests on validation set
   - **Cell 13-14**: Visualization - Generates confusion matrix and performance charts
   - **Cell 15**: Save Results - Exports final metrics to JSON

4. **Monitor Training:**
   - Watch the progress bars for epoch completion
   - Check training/validation loss and accuracy
   - Training will automatically save the best model

### Option 2: Using JupyterLab

```bash
# Install JupyterLab if not already installed
pip install jupyterlab

# Launch JupyterLab
jupyter lab

# Open DFER_GCViT_Mac.ipynb
```

### Option 3: Using VS Code

1. Install Python extension in VS Code
2. Open the notebook file
3. Select your Python interpreter (from venv)
4. Run cells using Shift+Enter

---

## 🧠 Model Architecture

### GCViT (Global Context Vision Transformer)

The model uses a hierarchical vision transformer architecture with global context modeling:

**Architecture Components:**
- **Backbone**: GCViT-xxtiny pre-trained on ImageNet-1K
- **Input Resolution**: 224×224 pixels
- **Feature Extraction**: 4 stages with increasing channels (64→128→256→512)
- **Global Context**: Self-attention with window-based and global tokens
- **Classification Head**: Custom fully connected layer (512→7 classes)

**Model Statistics:**
- Total Parameters: ~50.5M
- Trainable Parameters: ~50.5M (Full fine-tuning)
- Input Channels: 3 (RGB)
- Output Classes: 7 (Facial expressions)

**Key Features:**
- Hierarchical feature extraction
- Local-global context modeling
- Efficient attention mechanisms
- Pre-trained weights from ImageNet

---

## ⚙️ Training Configuration

### Hyperparameters

```python
BATCH_SIZE = 32              # Training batch size
LEARNING_RATE = 1e-4         # Initial learning rate
WEIGHT_DECAY = 1e-4          # L2 regularization
NUM_EPOCHS = 50              # Total training epochs
NUM_WORKERS = 4              # DataLoader workers
GRADIENT_CLIP = 1.0          # Gradient clipping threshold
```

### Data Augmentation

**Training Augmentation:**
- Random horizontal flip (p=0.5)
- Random brightness/contrast adjustment (p=0.3)
- Random rotation (±15 degrees, p=0.3)
- Gaussian blur (p=0.2)
- Normalization (ImageNet statistics)

**Validation Augmentation:**
- Resize to 224×224
- Normalization only

### Optimization Strategy

- **Optimizer**: AdamW with weight decay
- **Loss Function**: CrossEntropyLoss
- **LR Scheduler**: CosineAnnealingLR (T_max=50)
- **Early Stopping**: Best model saved based on validation accuracy
- **Mixed Precision**: Optional FP16 training

### Fine-Tuning Strategy

**Full Fine-Tuning (Current Implementation):**
- All model parameters are trainable
- Updates both backbone and classification head
- Higher computational cost but best performance
- Suitable when sufficient data is available

---

## 📈 Results

### Expected Performance Metrics

Based on the full fine-tuning strategy:

```
Test Accuracy:  ~71-72%
Test F1-Score:  ~71-72%
Total Params:   50,522,348
Training Time:  ~23 minutes/epoch (M3 Pro)
```

### Output Files

After training completes, you'll find:

1. **best_model.pth** - Trained model weights
2. **confusion_matrix.png** - Class-wise prediction analysis
3. **performance_summary.png** - Metrics visualization
4. **final_results.json** - Detailed metrics and statistics
5. **training_curve.png** - Loss and accuracy plots

### Interpreting Results

**Confusion Matrix:**
- Diagonal elements show correct predictions
- Off-diagonal elements show misclassifications
- Useful for identifying which classes are confused

**Performance Metrics:**
- **Accuracy**: Overall correctness
- **Precision**: Reliability of positive predictions
- **Recall**: Ability to find all positives
- **F1-Score**: Harmonic mean of precision and recall

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. ModuleNotFoundError: No module named 'torch'

**Problem:** Dependencies not installed

**Solution:**
```bash
pip install torch torchvision timm albumentations h5py opencv-python-headless tqdm pandas matplotlib seaborn scikit-learn
```

#### 2. MPS device not available

**Problem:** Apple Silicon GPU not detected

**Solution:**
- Ensure you're using macOS 12.3+
- Update PyTorch: `pip install --upgrade torch torchvision`
- Check MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`

#### 3. FileNotFoundError: archive.zip not found

**Problem:** Dataset not in correct location

**Solution:**
- Verify `archive.zip` is in the `WORKING_DIR` path
- Update the `WORKING_DIR` variable in Cell 4
- Ensure path uses forward slashes, even on Windows

#### 4. CUDA out of memory (if using NVIDIA GPU)

**Problem:** Batch size too large for GPU memory

**Solution:**
- Reduce `BATCH_SIZE` to 16 or 8
- Enable gradient checkpointing
- Use mixed precision training

#### 5. Training is very slow

**Problem:** Not using GPU acceleration

**Solution:**
- Check device in Cell 3: Should show "mps" (Mac) or "cuda" (NVIDIA)
- Install correct PyTorch version for your hardware
- Reduce `NUM_WORKERS` if CPU is bottleneck

#### 6. High memory usage during preprocessing

**Problem:** All images loaded into memory at once

**Solution:**
- Close other applications
- Restart kernel: `Kernel → Restart Kernel`
- Use smaller subset for testing

#### 7. Import error for albumentations

**Problem:** Albumentations version incompatibility

**Solution:**
```bash
pip install --upgrade albumentations opencv-python-headless
```

#### 8. Kernel dies during training

**Problem:** Insufficient memory

**Solution:**
- Reduce batch size to 16
- Close browser tabs and other applications
- Monitor memory usage: `htop` (Linux/Mac) or Task Manager (Windows)

---

## 🎯 Next Steps

### After Successful Training

1. **Analyze Results:**
   - Review confusion matrix for class-specific performance
   - Check which expressions are most confused
   - Analyze per-class precision and recall

2. **Model Deployment:**
   - Export model to ONNX format for production
   - Optimize for inference (TorchScript, quantization)
   - Test on real-world driving scenarios

3. **Further Experiments:**
   - Try partial fine-tuning (freeze backbone layers)
   - Experiment with different learning rates
   - Test data augmentation strategies
   - Collect more diverse training data

4. **Research Extensions:**
   - Multi-task learning (expression + engagement level)
   - Temporal modeling for video sequences
   - Real-time inference optimization
   - Cross-dataset generalization testing

---

## 📚 Additional Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [timm Library](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/docs/)
- [GCViT Paper](https://arxiv.org/abs/2206.09959)

### Related Work
- Vision Transformers (ViT)
- Facial Expression Recognition in the Wild
- Driver Monitoring Systems
- Transfer Learning for Computer Vision

---

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@misc{masood2025dfergcvit,
  author = {Masood, Syed Faizan Abbas},
  title = {DFER-GCViT: Driver Facial Expression Recognition using Global Context Vision Transformer},
  year = {2025},
  institution = {BTU Cottbus-Senftenberg},
  department = {Department of Graphical Systems}
}
```

---

## 🤝 Support

For questions or issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review error messages carefully
3. Contact: Syed Faizan Abbas Masood
4. Institution: BTU Cottbus-Senftenberg

---

## 📄 License

This project is for academic research purposes.

---

## ✅ Quick Start Checklist

Before running the project, ensure you have:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] KMU-FED dataset downloaded (`archive.zip`)
- [ ] Dataset placed in correct directory
- [ ] `WORKING_DIR` updated in notebook
- [ ] Jupyter Notebook installed and launched
- [ ] GPU (MPS/CUDA) detected (optional but recommended)

---

**Ready to start? Launch Jupyter Notebook and open `DFER_GCViT_Mac.ipynb`! 🚀**

---

*Last Updated: January 2025*  
*Author: Syed Faizan Abbas Masood*  
*Institution: BTU Cottbus-Senftenberg*
