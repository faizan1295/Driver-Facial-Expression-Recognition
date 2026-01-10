# 🚗 Driver Facial Expression Recognition using GCViT

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Efficient Fine-Tuning Strategies for Real-Time Automotive Applications using Global Context Vision Transformer (GCViT)**

## 📋 Overview

This research project investigates the effectiveness of partial fine-tuning strategies compared to full fine-tuning for driver facial expression recognition. The goal is to develop computationally efficient models suitable for real-time automotive safety applications while maintaining high classification accuracy.

### Research Question

> *How does partial fine-tuning of the DFER-GCViT model compare to full fine-tuning in terms of classification performance and computational efficiency?*

## 🎯 Key Results

| Strategy | Accuracy | F1-Score | Trainable Params | Avg Epoch Time |
|----------|----------|----------|------------------|----------------|
| **Full Fine-Tuning** | **71.75%** | **0.7156** | 50.5M (100%) | ~1370s |
| Last 2 Blocks | 67.67% | 0.6745 | 6.1M (12.1%) | ~520s |

### Key Findings

- ✅ Full fine-tuning achieves **4% higher accuracy**
- ✅ Partial fine-tuning is **2.6x faster** per epoch
- ✅ Partial fine-tuning uses **88% fewer trainable parameters**
- ✅ Both strategies effectively handle class imbalance

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

## 📊 Dataset

**KMU-FED (Keimyung University Facial Expression Dataset)**

| Split | Images | Description |
|-------|--------|-------------|
| Training | 28,709 | Used for model training |
| Test | 7,178 | Used for evaluation |
| **Total** | **35,887** | 7 emotion classes |

### Emotion Classes

| Class | Label | Training Samples |
|-------|-------|------------------|
| Angry | 0 | ~4,000 |
| Disgust | 1 | ~500 (minority) |
| Fear | 2 | ~4,100 |
| Happy | 3 | ~7,200 |
| Neutral | 4 | ~5,000 |
| Sad | 5 | ~4,800 |
| Surprise | 6 | ~3,100 |

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Apple Silicon Mac (MPS) or NVIDIA GPU (CUDA)

### Setup

```bash
# Clone the repository
git clone https://github.com/faizan1295/Driver-Facial-Expression-Recognition.git
cd Driver-Facial-Expression-Recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision timm albumentations h5py opencv-python-headless tqdm pandas matplotlib seaborn scikit-learn
```

## 🚀 Usage

### Training

Open the Jupyter notebook and run all cells:

```bash
jupyter notebook DFER_GCViT_Mac.ipynb
```

### Configuration

Modify training parameters in the notebook:

```python
TRAINING_CONFIG = {
    'epochs': 60,
    'learning_rate': 0.0001,
    'weight_decay': 1e-4,
    'patience': 30,
    'min_delta': 0.001
}
```

### Fine-Tuning Strategies

```python
# Strategy 1: Full Fine-Tuning
model = DFER_GCViT(pretrained=True)
# All parameters trainable

# Strategy 2: Partial Fine-Tuning (Last 2 Blocks)
model = DFER_GCViT(pretrained=True)
model.freeze_all_except_last_n_blocks(n_blocks=2)
```

## 📁 Project Structure

```
Driver-Facial-Expression-Recognition/
├── DFER_GCViT_Mac.ipynb        # Main notebook
├── README.md                    # This file
├── Project_Report.md            # Detailed research report
├── training_results.json        # Saved metrics
├── kmu_fed_config.json          # Dataset configuration
├── comparison_results.png       # Performance comparison chart
├── loss_curves.png              # Training/validation loss curves
├── confusion_matrices.png       # Per-class performance
└── .gitignore                   # Git ignore rules
```

## 📈 Training Visualizations

### Loss Curves

The training demonstrates effective learning with early stopping preventing overfitting:

- **Full Fine-Tuning**: Converged at epoch 40
- **Partial Fine-Tuning**: Converged at epoch 41

### Performance Comparison

| Metric | Full Fine-Tuning | Partial (Last 2 Blocks) | Difference |
|--------|------------------|-------------------------|------------|
| Accuracy | 71.75% | 67.67% | +4.08% |
| Precision | 71.55% | 67.41% | +4.14% |
| Recall | 71.75% | 67.67% | +4.08% |
| F1-Score | 71.56% | 67.45% | +4.11% |
| Training Time | ~15.2 hrs | ~5.9 hrs | 2.6x faster |

## 🔧 Technical Details

### Hardware

- **Device**: MacBook Pro with Apple M3 Pro
- **Acceleration**: Metal Performance Shaders (MPS)
- **Memory**: Optimized batch size (16) for MPS compatibility

### Software Stack

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.8.0 | Deep learning framework |
| timm | 1.0.22 | Pre-trained GCViT model |
| Albumentations | Latest | Data augmentation |
| scikit-learn | Latest | Metrics calculation |

### Data Augmentation

```python
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

## 📚 References

1. Hatamizadeh, A., et al. (2023). "Global Context Vision Transformers." ICML 2023.
2. Goodfellow, I. J., et al. (2013). "Challenges in representation learning: FER2013."
3. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: ViT." ICLR 2021.
4. Liu, Z., et al. (2021). "Swin Transformer." ICCV 2021.
5. Mollahosseini, A., et al. (2017). "AffectNet: FER in the Wild." IEEE TAC.

## 👤 Author

**Syed Faizan Abbas Masood**  
Masters of Artificial Intelligence
BTU Cottbus-Senftenberg  

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BTU Cottbus-Senftenberg for research support
- Keimyung University for the KMU-FED dataset
- timm library maintainers for pre-trained models
- PyTorch team for the deep learning framework

---

*For questions or collaborations, please open an issue or contact the author.*