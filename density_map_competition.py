import h5py
import json
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

def generate_density_maps_competition(dataset_root):
    """
    Generate density maps and attention maps from competition JSON format.
    Compatible with existing SFANet pipeline.
    
    Args:
        dataset_root: Path to dataset folder containing train/ and test/ subdirs
    """
    
    # Define paths
    train_images_path = os.path.join(dataset_root, 'train', 'images')
    train_labels_path = os.path.join(dataset_root, 'train', 'labels')
    
    # Create output directory for processed data
    output_path = os.path.join(dataset_root, 'train', 'new_data')
    os.makedirs(output_path, exist_ok=True)
    
    # Get all image files
    image_files = glob.glob(os.path.join(train_images_path, '*.jpg'))
    image_files.sort()
    
    print(f"Found {len(image_files)} training images")
    
    processed_count = 0
    error_count = 0
    
    for img_path in image_files:
        try:
            # Get corresponding JSON file
            img_name = os.path.basename(img_path)
            json_name = img_name.replace('.jpg', '.json')
            json_path = os.path.join(train_labels_path, json_name)
            
            if not os.path.exists(json_path):
                print(f"Warning: No label file found for {img_name}")
                continue
                
            # Load image to get dimensions
            img = Image.open(img_path)
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            
            # Load JSON annotations
            with open(json_path, 'r') as f:
                label_data = json.load(f)
            
            # Extract points and human_num
            points = label_data['points']
            human_num_label = label_data['human_num']
            
            # Initialize density map
            density_map = np.zeros((height, width), dtype=np.float32)
            
            # Process each coordinate point
            valid_points = 0
            invalid_points = 0
            
            for point in points:
                # Handle different JSON formats
                if isinstance(point, dict):
                    x, y = point.get('x', 0), point.get('y', 0)
                else:
                    # Handle list format [x, y] 
                    x, y = point[0], point[1]
                
                # Convert to integer coordinates with proper rounding
                x_int = int(round(x))
                y_int = int(round(y))
                
                # Boundary checking
                if 0 <= y_int < height and 0 <= x_int < width:
                    density_map[y_int, x_int] = 1.0
                    valid_points += 1
                else:
                    invalid_points += 1
                    print(f"  Out-of-bounds coordinate in {img_name}: ({x}, {y}) -> ({x_int}, {y_int}), image size: {width}x{height}")
            
            # Apply Gaussian filter (same as original - sigma=5)
            density_map = gaussian_filter(density_map, 5)
            
            # Generate attention map (same threshold as original)
            attention_map = (density_map > 0.001).astype(np.float32)
            
            # Verify count consistency
            actual_count = valid_points
            if actual_count != human_num_label:
                print(f"  Count mismatch in {img_name}: JSON says {human_num_label}, found {actual_count} valid points")
            
            # Save as .h5 file (compatible with existing dataset.py)
            h5_filename = img_name.replace('.jpg', '.h5')
            h5_path = os.path.join(output_path, h5_filename)
            
            with h5py.File(h5_path, 'w') as hf:
                hf['density'] = density_map
                hf['attention'] = attention_map
                hf['gt'] = actual_count  # Use actual valid points count
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{len(image_files)} images")
                
            # Log statistics for first few images
            if processed_count <= 5:
                print(f"Image {img_name}:")
                print(f"  Size: {width}x{height}")
                print(f"  Valid points: {valid_points}, Invalid: {invalid_points}")
                print(f"  Density sum: {density_map.sum():.2f}")
                print(f"  Attention pixels: {attention_map.sum()}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            error_count += 1
            continue
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output saved to: {output_path}")

def visualize_sample_results(dataset_root, num_samples=3):
    """
    Visualize sample density maps to verify generation quality
    """
    output_path = os.path.join(dataset_root, 'train', 'new_data')
    images_path = os.path.join(dataset_root, 'train', 'images')
    
    h5_files = glob.glob(os.path.join(output_path, '*.h5'))[:num_samples]
    
    for h5_path in h5_files:
        # Load processed data
        with h5py.File(h5_path, 'r') as hf:
            density = np.array(hf['density'])
            attention = np.array(hf['attention'])
            gt_count = np.array(hf['gt'])
        
        # Load original image
        h5_filename = os.path.basename(h5_path)
        img_filename = h5_filename.replace('.h5', '.jpg')
        img_path = os.path.join(images_path, img_filename)
        img = plt.imread(img_path)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title(f'Original Image\n{img_filename}')
        axes[0].axis('off')
        
        im1 = axes[1].imshow(density, cmap='jet')
        axes[1].set_title(f'Density Map\nSum: {density.sum():.1f}, GT: {gt_count}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        axes[2].imshow(attention, cmap='gray')
        axes[2].set_title(f'Attention Map\nActive pixels: {attention.sum():.0f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'density_visualization_{img_filename.replace(".jpg", ".png")}')
        plt.show()

if __name__ == '__main__':
    # Configuration
    dataset_root = './dataset'  # Adjust path as needed
    
    print("Starting competition dataset preprocessing...")
    print("Converting JSON annotations to density maps...")
    
    # Generate density maps
    generate_density_maps_competition(dataset_root)
    
    # Visualize results
    print("\nGenerating sample visualizations...")
    visualize_sample_results(dataset_root, num_samples=3)
    
    print("\nPreprocessing completed! Ready for training.")