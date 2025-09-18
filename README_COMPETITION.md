# SFANet Competition Setup Guide

This guide explains how to adapt the SFANet repository for crowd counting competition.

## 📋 Prerequisites

1. **Python Environment**:

   - Python 3.7+
   - PyTorch 1.1.0+
   - CUDA-capable GPU (recommended)

2. **Dependencies**:
   ```bash
   pip install torch torchvision
   pip install h5py scipy pillow opencv-python
   pip install tensorboardX matplotlib
   pip install pandas numpy tqdm
   ```

## 🗂️ Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── train/
│   ├── images/          # Training images (.jpg)
│   └── labels/          # JSON annotations
├── test/
│   └── images/          # Test images (.jpg)
└── sample_submission.csv
```

**JSON Format** (in `train/labels/`):

```json
{
    "img_id": "1.jpg",
    "human_num": 25,
    "points": [
        {"x": 120.5, "y": 72.7},
        {"x": 128.9, "y": 67.2},
        ...
    ]
}
```

## 🚀 Quick Start

### Step 1: Prepare Density Maps

Convert JSON annotations to density maps:

```bash
python density_map_competition.py
```

This will:

- Read JSON coordinates from `dataset/train/labels/`
- Generate Gaussian density maps (sigma=5)
- Create attention maps
- Save `.h5` files to `dataset/train/new_data/`

### Step 2: Train Model

Start training with competition data:

```bash
python train_competition.py --data_path ./dataset --save_path ./checkpoints_competition
```

Optional parameters:

- `--bs 8`: Batch size (default: 8)
- `--epoch 500`: Training epochs (default: 500)
- `--lr 1e-4`: Learning rate (default: 1e-4)
- `--gpu 0`: GPU ID (default: 0)
- `--load`: Resume from checkpoint

### Step 3: Generate Predictions

Create submission file:

```bash
python inference_competition.py --data_path ./dataset --save_path ./checkpoints_competition --output_path ./submission.csv
```

Parameters:

- `--checkpoint best`: Use best or latest checkpoint
- `--batch_size 1`: Batch size for inference
- `--gpu 0`: GPU ID

## 📁 New Files Created

### Core Implementation Files:

1. **`density_map_competition.py`**: Converts JSON → density maps
2. **`dataset_competition.py`**: Competition dataset loader
3. **`train_competition.py`**: Training script for competition
4. **`inference_competition.py`**: Inference and submission generation

### Generated Files:

- `dataset/train/new_data/*.h5`: Preprocessed density maps
- `checkpoints_competition/`: Model checkpoints
- `logs_competition/`: TensorBoard logs
- `submission.csv`: Final submission file

## 🔧 Configuration Options

### Model Architecture:

- **Preserved**: Original SFANet architecture (VGG + dual path + attention)
- **Input**: 400×400 for training, variable for testing (rounded to multiples of 16)
- **Output**: Density maps with stride=2

### Training Strategy:

- **Loss**: MSE (density) + BCE (attention, weight=0.1)
- **Optimizer**: Adam with lr=1e-4
- **Augmentation**: Random resize, crop, flip, gamma, grayscale
- **Validation**: 20% split from training data

### Data Processing:

- **Coordinate handling**: Float coordinates → integer with proper rounding
- **Boundary checking**: Out-of-bounds coordinates filtered
- **Density conservation**: Total count preserved during scaling
- **Gaussian filtering**: sigma=5 (same as original)

## 📊 Expected Performance

Based on original SFANet results:

- **ShanghaiTech Part A**: MAE 60.43, MSE 98.24
- **ShanghaiTech Part B**: MAE 6.38, MSE 10.99

Competition performance will depend on dataset characteristics.

## 🐛 Troubleshooting

### Common Issues:

1. **Out-of-bounds coordinates**:

   - Solution: Already handled in `density_map_competition.py`
   - Effect: Invalid coordinates are filtered out

2. **Memory issues**:

   - Reduce batch size: `--bs 4`
   - Use CPU for preprocessing: Remove `.cuda()` calls

3. **CUDA errors**:

   - Check GPU availability: `torch.cuda.is_available()`
   - Verify GPU ID: `--gpu 0`

4. **Import errors**:
   - Install missing packages: `pip install <package>`
   - Check Python environment activation

### Validation Steps:

1. **Check density map generation**:

   ```python
   # Run density_map_competition.py and check visualizations
   ```

2. **Test dataset loader**:

   ```python
   # Run dataset_competition.py to validate data loading
   ```

3. **Monitor training**:

   ```bash
   tensorboard --logdir logs_competition
   ```

4. **Validate submission format**:
   ```bash
   python inference_competition.py --validate
   ```

## 🔍 Monitoring Training

### TensorBoard Metrics:

- `loss/train_loss_density`: MSE loss for density maps
- `loss/train_loss_attention`: BCE loss for attention maps
- `eval/MAE`: Mean Absolute Error on validation
- `eval/MSE`: Mean Squared Error on validation

### Key Indicators:

- **Converging losses**: Both density and attention losses should decrease
- **Stable validation**: MAE should improve over epochs
- **Reasonable predictions**: Count predictions should be sensible

## 📈 Performance Optimization

### Training Speedup:

1. **Increase batch size** (if GPU memory allows)
2. **Use multiple workers**: `num_workers=4` in DataLoader
3. **Mixed precision**: Add `torch.cuda.amp` for faster training

### Accuracy Improvements:

1. **Data augmentation**: Tune augmentation parameters in `transforms.py`
2. **Learning rate scheduling**: Add lr scheduler
3. **Ensemble methods**: Combine multiple model predictions
4. **Post-processing**: Apply density map smoothing

## 📝 Submission Format

The final submission file (`submission.csv`) should contain:

```csv
image_id,predicted_count
1.jpg,25
2.jpg,143
...
```

**Important Notes**:

- Predictions are automatically rounded to integers
- Negative predictions are clipped to 0
- File is sorted by image_id for consistency

## 🎯 Next Steps

1. **Hyperparameter tuning**: Experiment with learning rates, batch sizes
2. **Architecture modifications**: Try different backbones or attention mechanisms
3. **Ensemble methods**: Combine multiple models for better accuracy
4. **Post-processing**: Apply domain-specific knowledge for refinement

## 📚 References

- Original SFANet paper: [Dual Path Multi-Scale Fusion Networks with Attention for Crowd Counting](https://arxiv.org/abs/1902.01115)
- Repository: [SFANet-crowd-counting](https://github.com/pxq0312/SFANet-crowd-counting)
