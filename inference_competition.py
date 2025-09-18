import torch
from torch.utils import data
from dataset_competition import DatasetCompetition
from models import Model
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./dataset', type=str, help='path to competition dataset')
    parser.add_argument('--save_path', default='./checkpoints_competition', type=str, help='path to saved checkpoint')
    parser.add_argument('--output_path', default='./submission.csv', type=str, help='path to save submission file')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for inference')
    parser.add_argument('--checkpoint', default='best', choices=['best', 'latest'], help='which checkpoint to use')
    
    args = parser.parse_args()
    
    print("=== SFANet Competition Inference ===")
    print(f"Dataset path: {args.data_path}")
    print(f"Checkpoint path: {args.save_path}")
    print(f"Output path: {args.output_path}")
    
    # Setup device
    device = torch.device('cuda:' + str(args.gpu))
    print(f"Using device: {device}")
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = DatasetCompetition(args.data_path, is_train=False, is_test=True)
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loader
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                  num_workers=2, pin_memory=True)
    
    # Create model
    model = Model().to(device)
    
    # Load checkpoint
    checkpoint_name = f'checkpoint_{args.checkpoint}.pth'
    checkpoint_path = os.path.join(args.save_path, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    if 'mae' in checkpoint:
        print(f"Checkpoint MAE: {checkpoint['mae']:.4f}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    # Set model to evaluation mode
    model.eval()
    
    print("\nRunning inference...")
    predictions = []
    image_ids = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tqdm(test_loader, desc="Processing")):
            images = images.to(device)
            
            # Forward pass
            density_output, attention_output = model(images)
            
            # Calculate predicted count
            predicted_counts = density_output.sum(dim=(1, 2, 3))  # Sum over spatial dimensions
            
            # Store results
            for i, filename in enumerate(filenames):
                pred_count = predicted_counts[i].item()
                predictions.append(pred_count)
                image_ids.append(filename)
                
                if batch_idx < 5:  # Print first few predictions
                    print(f"  {filename}: {pred_count:.2f}")
    
    inference_time = time.time() - start_time
    print(f"\nInference completed in {inference_time:.2f}s")
    print(f"Average time per image: {inference_time/len(test_dataset)*1000:.1f}ms")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'image_id': image_ids,
        'predicted_count': predictions
    })
    
    # Round predictions to integers (crowd counts should be whole numbers)
    submission_df['predicted_count'] = submission_df['predicted_count'].round().astype(int)
    
    # Ensure non-negative predictions
    submission_df['predicted_count'] = submission_df['predicted_count'].clip(lower=0)
    
    # Sort by image_id for consistent submission format
    submission_df = submission_df.sort_values('image_id').reset_index(drop=True)
    
    # Save submission file
    submission_df.to_csv(args.output_path, index=False)
    print(f"\nSubmission saved to: {args.output_path}")
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"  Total images: {len(predictions)}")
    print(f"  Min prediction: {submission_df['predicted_count'].min()}")
    print(f"  Max prediction: {submission_df['predicted_count'].max()}")
    print(f"  Mean prediction: {submission_df['predicted_count'].mean():.2f}")
    print(f"  Median prediction: {submission_df['predicted_count'].median():.2f}")
    
    # Show distribution
    print(f"\nPrediction Distribution:")
    bins = [0, 10, 50, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['0-10', '11-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
    
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if high == float('inf'):
            count = (submission_df['predicted_count'] >= low).sum()
        else:
            count = ((submission_df['predicted_count'] >= low) & 
                    (submission_df['predicted_count'] <= high)).sum()
        percentage = count / len(submission_df) * 100
        print(f"  {bin_labels[i]}: {count} images ({percentage:.1f}%)")
    
    # Preview submission file
    print(f"\nSubmission Preview:")
    print(submission_df.head(10))
    
    print("\nInference completed successfully!")

def validate_submission_format(csv_path, sample_csv_path):
    """
    Validate that the submission file matches the expected format
    """
    print(f"\nValidating submission format...")
    
    # Load files
    submission = pd.read_csv(csv_path)
    sample = pd.read_csv(sample_csv_path)
    
    print(f"Submission shape: {submission.shape}")
    print(f"Sample shape: {sample.shape}")
    
    # Check columns
    if list(submission.columns) != list(sample.columns):
        print(f"WARNING: Column mismatch!")
        print(f"  Expected: {list(sample.columns)}")
        print(f"  Got: {list(submission.columns)}")
        return False
    
    # Check image IDs
    submission_ids = set(submission['image_id'])
    sample_ids = set(sample['image_id'])
    
    missing_ids = sample_ids - submission_ids
    extra_ids = submission_ids - sample_ids
    
    if missing_ids:
        print(f"WARNING: Missing {len(missing_ids)} image IDs")
        if len(missing_ids) <= 5:
            print(f"  Missing: {missing_ids}")
    
    if extra_ids:
        print(f"WARNING: Extra {len(extra_ids)} image IDs")
        if len(extra_ids) <= 5:
            print(f"  Extra: {extra_ids}")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(submission['predicted_count']):
        print(f"WARNING: predicted_count should be numeric!")
        return False
    
    print("Submission format validation completed!")
    return len(missing_ids) == 0 and len(extra_ids) == 0

if __name__ == '__main__':
    main()
    
    # Optionally validate submission format
    import sys
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if '--validate' in args:
            sample_csv = './dataset/sample_submission.csv'
            submission_csv = './submission.csv'
            
            if os.path.exists(sample_csv) and os.path.exists(submission_csv):
                validate_submission_format(submission_csv, sample_csv)
            else:
                print("Sample submission or generated submission not found for validation")