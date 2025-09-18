# 🎯 SFANet Crowd Counting - Competition Ready

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/SFANet-crowd-counting/blob/main/colab_setup.ipynb)

This repository contains an implementation of SFANet ([Dual Path Multi-Scale Fusion Networks with Attention for Crowd Counting](https://arxiv.org/abs/1902.01115)) adapted for crowd counting competitions with JSON annotation format.

## 🚀 Quick Start on Google Colab

### 1. Clone Repository
```bash
!git clone https://github.com/yourusername/SFANet-crowd-counting.git
%cd SFANet-crowd-counting
```

### 2. Install Dependencies
```bash
!pip install torch torchvision torchaudio
!pip install h5py scipy pillow matplotlib tensorboard
```

### 3. Upload Your Dataset
```python
# Upload your competition dataset to ./dataset/
# Structure should be:
# dataset/
#   train/
#     images/
#     labels/
#   test/
#     images/
#   sample_submission.csv
```

### 4. Run Training Pipeline
```bash
# Preprocess dataset (JSON → density maps)
!python density_map_competition.py

# Start training
!python train_competition.py --data_path ./dataset --save_path ./checkpoints

# Generate submission
!python inference_competition.py --save_path ./checkpoints --output_path ./submission.csv
```

## 📋 Prerequisites

- **Python**: 3.7+
- **PyTorch**: 1.1.0+
- **CUDA**: Recommended for training
- **Memory**: 8GB+ RAM, 4GB+ GPU memory

## 🗂️ Repository Structure

### Core Files
- `models.py` - SFANet architecture (VGG backbone + dual path)
- `transforms.py` - Data augmentation pipeline
- `train_competition.py` - Training script for competition format
- `inference_competition.py` - Generate competition submission
- `dataset_competition.py` - Competition dataset loader
- `density_map_competition.py` - JSON to density map converter

### Documentation
- `README_COMPETITION.md` - Detailed setup guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `test_pipeline.py` - End-to-end testing script

### Original Files (for reference)
- `train.py` - Original training script
- `eval.py` - Original evaluation script
- `dataset.py` - Original dataset loader
- `density_map.py` - Original density map generator

## 🎯 Competition Usage

### Training Pipeline
```bash
# 1. Convert JSON annotations to density maps
python density_map_competition.py

# 2. Train model with competition data
python train_competition.py --data_path ./dataset --save_path ./checkpoints_competition

# 3. Generate submission file
python inference_competition.py --save_path ./checkpoints_competition --output_path ./submission.csv
```

### Key Features
- ✅ **JSON Format Support**: Handles float coordinates from competition data
- ✅ **Variable Image Sizes**: Supports different resolutions (360×640 to 1920×1080)
- ✅ **Robust Preprocessing**: Boundary checking and coordinate validation
- ✅ **Density Conservation**: Maintains accurate crowd counts during scaling
- ✅ **Competition Ready**: Direct CSV output for submission

## 📊 Original Results

**ShanghaiTech Part A**: MAE 60.43, MSE 98.24  
**ShanghaiTech Part B**: MAE 6.38, MSE 10.99

![Results A](./logs/A.png)
![Results B](./logs/B.png)

## 🔧 Configuration

### Training Parameters
- **Batch Size**: 8 (adjust based on GPU memory)
- **Learning Rate**: 1e-4 with Adam optimizer
- **Input Size**: 400×400 (training), variable (testing)
- **Loss Function**: MSE (density) + BCE (attention)

### Data Processing
- **Gaussian Kernel**: σ=5 for density map generation
- **Attention Threshold**: 0.001 for attention map creation
- **Augmentation**: Random resize, crop, flip, gamma, grayscale

## 🐛 Troubleshooting

See [README_COMPETITION.md](README_COMPETITION.md) for detailed troubleshooting guide.

## 📝 License

This project is based on the original SFANet paper. Please cite the original work:

```bibtex
@article{zhu2019dual,
  title={Dual path multi-scale fusion networks with attention for crowd counting},
  author={Zhu, Liang and Zhao, Zhijian and Lu, Chao and Lin, Yao and Peng, Yanning and Yao, Tangren},
  journal={arXiv preprint arXiv:1902.01115},
  year={2019}
}
```
