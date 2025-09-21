#!/usr/bin/env python3
"""
Simplified TTA inference script for SFANet crowd counting.
This script provides an easy way to run test time augmentation.
"""

import torch
import torch.nn.functional as F
from torch.utils import data
from dataset_competition import DatasetCompetition
from models import Model
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import argparse

def inference_with_tta(model, image, device, tta_transforms=['original', 'hflip']):
    """
    Perform test time augmentation on a single image
    
    Args:
        model: trained model
        image: input tensor [1, 3, H, W]
        device: cuda device
        tta_transforms: list of augmentations to apply
    
    Returns:
        averaged prediction count
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            if transform == 'original':
                # Original image
                aug_image = image
            elif transform == 'hflip':
                # Horizontal flip
                aug_image = torch.flip(image, dims=[3])  # Flip width dimension
            elif transform == 'scale_up':
                # Scale up 10%
                _, _, h, w = image.shape
                new_h, new_w = int(h * 1.1), int(w * 1.1)
                # Round to multiple of 16
                new_h = round(new_h / 16) * 16
                new_w = round(new_w / 16) * 16
                aug_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
            elif transform == 'scale_down':
                # Scale down 10%
                _, _, h, w = image.shape
                new_h, new_w = int(h * 0.9), int(w * 0.9)
                # Round to multiple of 16
                new_h = round(new_h / 16) * 16
                new_w = round(new_w / 16) * 16
                aug_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Forward pass
            density_output, _ = model(aug_image)
            
            # Handle density map scaling for resized images
            if transform in ['scale_up', 'scale_down']:
                # Resize density map back to original scale for consistent counting
                _, _, orig_h, orig_w = image.shape
                # Density map is stride=2, so half the input size
                target_h, target_w = orig_h // 2, orig_w // 2
                density_output = F.interpolate(density_output, size=(target_h, target_w), 
                                             mode='bilinear', align_corners=False)
            
            # Calculate count
            pred_count = density_output.sum().item()
            predictions.append(pred_count)
    
    # Return averaged prediction
    return np.mean(predictions)

def run_tta_inference(data_path='./dataset', 
                     checkpoint_path='./checkpoints_competition/checkpoint_best.pth',
                     output_path='./submission_tta.csv',
                     tta_type='fast',
                     gpu_id=0):
    """
    Run TTA inference with predefined configurations
    
    Args:
        data_path: path to dataset
        checkpoint_path: path to trained model
        output_path: path to save submission
        tta_type: 'fast', 'balanced', or 'custom'
        gpu_id: GPU device ID
    """
    
    # Define TTA configurations
    tta_configs = {
        'fast': ['original', 'hflip'],  # 2x time, good accuracy boost
        'balanced': ['original', 'hflip', 'scale_up', 'scale_down'],  # 4x time, better accuracy
        'minimal': ['original'],  # 1x time, no TTA (baseline)
    }
    
    if tta_type not in tta_configs:
        raise ValueError(f"Unknown TTA type: {tta_type}. Choose from {list(tta_configs.keys())}")
    
    tta_transforms = tta_configs[tta_type]
    
    print("🚀 SFANet TTA Inference")
    print(f"📁 Dataset: {data_path}")
    print(f"💾 Checkpoint: {checkpoint_path}")
    print(f"📄 Output: {output_path}")
    print(f"🔧 TTA Type: {tta_type}")
    print(f"🎯 Transforms: {tta_transforms}")
    print(f"⏱️  Expected time multiplier: {len(tta_transforms)}x")
    
    # Setup device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load dataset
    test_dataset = DatasetCompetition(data_path, is_train=False, is_test=True)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"🔢 Test samples: {len(test_dataset)}")
    
    # Load model
    model = Model().to(device)
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"✅ Model loaded (MAE: {checkpoint.get('mae', 'N/A')})")
    
    # Run inference
    model.eval()
    predictions = []
    image_ids = []
    
    start_time = time.time()
    
    print(f"\n🔄 Running {tta_type} TTA inference...")
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tqdm(test_loader, desc="Processing")):
            images = images.to(device)
            
            for i, filename in enumerate(filenames):
                single_image = images[i:i+1]  # Keep batch dimension
                pred_count = inference_with_tta(model, single_image, device, tta_transforms)
                
                predictions.append(pred_count)
                image_ids.append(filename)
                
                # Show progress for first few
                if batch_idx < 3:
                    print(f"  📸 {filename}: {pred_count:.1f}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Inference completed in {total_time:.1f}s")
    print(f"📊 Average time per image: {total_time/len(test_dataset):.2f}s")
    
    # Save results
    submission_df = pd.DataFrame({
        'image_id': image_ids,
        'predicted_count': np.round(predictions).astype(int).clip(min=0)
    })
    
    submission_df = submission_df.sort_values('image_id').reset_index(drop=True)
    submission_df.to_csv(output_path, index=False)
    
    # Print statistics
    print(f"\n📈 Results Summary:")
    print(f"  💾 Saved to: {output_path}")
    print(f"  🔢 Total predictions: {len(predictions)}")
    print(f"  📊 Mean count: {np.mean(predictions):.1f}")
    print(f"  📊 Std count: {np.std(predictions):.1f}")
    print(f"  📊 Min/Max: {np.min(predictions):.0f} / {np.max(predictions):.0f}")
    
    return submission_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFANet TTA Inference')
    parser.add_argument('--data_path', default='./dataset', help='Dataset path')
    parser.add_argument('--checkpoint', default='./checkpoints_competition/checkpoint_best.pth', help='Model checkpoint')
    parser.add_argument('--output', default='./submission_tta.csv', help='Output CSV file')
    parser.add_argument('--tta_type', default='fast', choices=['minimal', 'fast', 'balanced'], 
                        help='TTA configuration')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
    
    args = parser.parse_args()
    
    try:
        result = run_tta_inference(
            data_path=args.data_path,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            tta_type=args.tta_type,
            gpu_id=args.gpu
        )
        print("\n🎉 TTA inference completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during TTA inference: {e}")
        import traceback
        traceback.print_exc()