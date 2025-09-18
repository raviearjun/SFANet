from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from transforms import Transforms
import glob
from torchvision.transforms import functional


class DatasetCompetition(data.Dataset):
    """
    Competition dataset loader compatible with existing SFANet pipeline.
    Handles both training (with .h5 density maps) and test (images only) data.
    """
    def __init__(self, data_path, is_train, is_test=False):
        """
        Args:
            data_path: Path to dataset root (should contain 'train/' and 'test/' folders)
            is_train: Whether to use train split (True) or test split (False) 
            is_test: Whether this is for final test inference (affects data loading)
        """
        self.is_train = is_train
        self.is_test = is_test
        self.data_path = data_path
        
        if is_train:
            # Training data - load images and corresponding .h5 files
            images_path = os.path.join(data_path, 'train', 'images')
            labels_path = os.path.join(data_path, 'train', 'new_data')
            
            self.image_list = glob.glob(os.path.join(images_path, '*.jpg'))
            self.label_list = []
            
            # Find corresponding .h5 files for each image
            for img_path in self.image_list:
                img_name = os.path.basename(img_path)
                h5_name = img_name.replace('.jpg', '.h5')
                h5_path = os.path.join(labels_path, h5_name)
                
                if os.path.exists(h5_path):
                    self.label_list.append(h5_path)
                else:
                    # Remove image if no corresponding .h5 file
                    self.image_list.remove(img_path)
                    print(f"Warning: No .h5 file found for {img_name}, skipping")
            
            print(f"Training dataset: {len(self.image_list)} images with density maps")
            
        else:
            # Test data - either validation split from train or final test images
            if is_test:
                # Final test set (no labels available)
                images_path = os.path.join(data_path, 'test', 'images')
                self.image_list = glob.glob(os.path.join(images_path, '*.jpg'))
                self.label_list = []
                print(f"Test dataset: {len(self.image_list)} images (no labels)")
            else:
                # Validation split from training data
                images_path = os.path.join(data_path, 'train', 'images')
                labels_path = os.path.join(data_path, 'train', 'new_data')
                
                all_images = glob.glob(os.path.join(images_path, '*.jpg'))
                all_images.sort()
                
                # Use last 20% as validation set
                val_start = int(0.8 * len(all_images))
                self.image_list = all_images[val_start:]
                
                self.label_list = []
                for img_path in self.image_list:
                    img_name = os.path.basename(img_path)
                    h5_name = img_name.replace('.jpg', '.h5')
                    h5_path = os.path.join(labels_path, h5_name)
                    
                    if os.path.exists(h5_path):
                        self.label_list.append(h5_path)
                    else:
                        self.image_list.remove(img_path)
                        print(f"Warning: No .h5 file found for {img_name}, skipping")
                
                print(f"Validation dataset: {len(self.image_list)} images")
        
        # Sort lists to ensure consistent ordering
        self.image_list.sort()
        if self.label_list:
            self.label_list.sort()
    
    def __getitem__(self, index):
        # Load image
        image = Image.open(self.image_list[index]).convert('RGB')
        
        if self.is_test and not self.label_list:
            # Test mode - return image only with preprocessing
            height, width = image.size[1], image.size[0]
            # Round to multiple of 16 (required by model)
            height = round(height / 16) * 16
            width = round(width / 16) * 16
            image = image.resize((width, height), Image.BILINEAR)

            image = functional.to_tensor(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            # Return image and image filename for submission generation
            img_filename = os.path.basename(self.image_list[index])
            return image, img_filename
        
        else:
            # Training or validation mode - load density maps
            label = h5py.File(self.label_list[index], 'r')
            density = np.array(label['density'], dtype=np.float32)
            attention = np.array(label['attention'], dtype=np.float32)
            gt = np.array(label['gt'], dtype=np.float32)
            label.close()
            
            if self.is_train:
                # Training mode - apply data augmentation
                # Use same transform parameters as original
                trans = Transforms((0.8, 1.2), (400, 400), 2, (0.5, 1.5), 'SHA')
                image, density, attention = trans(image, density, attention)
                return image, density, attention
            else:
                # Validation mode - minimal preprocessing
                height, width = image.size[1], image.size[0]
                # Round to multiple of 16 (required by model)
                height = round(height / 16) * 16
                width = round(width / 16) * 16
                image = image.resize((width, height), Image.BILINEAR)

                image = functional.to_tensor(image)
                image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                return image, gt

    def __len__(self):
        return len(self.image_list)


def create_train_val_split(data_path, train_ratio=0.8):
    """
    Create train/validation split from competition training data.
    
    Args:
        data_path: Path to dataset root
        train_ratio: Fraction of data to use for training
    
    Returns:
        train_dataset, val_dataset
    """
    # Training dataset (first 80% of data)
    train_dataset = DatasetCompetition(data_path, is_train=True, is_test=False)
    
    # Validation dataset (last 20% of data) 
    val_dataset = DatasetCompetition(data_path, is_train=False, is_test=False)
    
    return train_dataset, val_dataset


def create_test_dataset(data_path):
    """
    Create test dataset for final inference.
    
    Args:
        data_path: Path to dataset root
        
    Returns:
        test_dataset
    """
    return DatasetCompetition(data_path, is_train=False, is_test=True)


if __name__ == '__main__':
    # Test the dataset loader
    data_path = './dataset'
    
    print("Testing competition dataset loader...")
    
    # Test training dataset
    print("\n=== Training Dataset ===")
    train_dataset = DatasetCompetition(data_path, is_train=True)
    print(f"Training samples: {len(train_dataset)}")
    
    if len(train_dataset) > 0:
        image, density, attention = train_dataset[0]
        print(f"Training sample shapes:")
        print(f"  Image: {image.shape}")
        print(f"  Density: {density.shape}")
        print(f"  Attention: {attention.shape}")
        print(f"  Density sum: {density.sum():.2f}")
    
    # Test validation dataset
    print("\n=== Validation Dataset ===")
    val_dataset = DatasetCompetition(data_path, is_train=False, is_test=False)
    print(f"Validation samples: {len(val_dataset)}")
    
    if len(val_dataset) > 0:
        image, gt = val_dataset[0]
        print(f"Validation sample shapes:")
        print(f"  Image: {image.shape}")
        print(f"  Ground truth count: {gt}")
    
    # Test test dataset
    print("\n=== Test Dataset ===")
    test_dataset = DatasetCompetition(data_path, is_train=False, is_test=True)
    print(f"Test samples: {len(test_dataset)}")
    
    if len(test_dataset) > 0:
        image, filename = test_dataset[0]
        print(f"Test sample:")
        print(f"  Image shape: {image.shape}")
        print(f"  Filename: {filename}")
    
    # Test data loaders
    print("\n=== Testing DataLoader compatibility ===")
    from torch.utils import data
    
    train_loader = data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Test one batch from each loader
    print("Training batch:")
    for batch in train_loader:
        images, densities, attentions = batch
        print(f"  Batch images: {images.shape}")
        print(f"  Batch densities: {densities.shape}")
        print(f"  Batch attentions: {attentions.shape}")
        break
    
    print("Validation batch:")
    for batch in val_loader:
        images, gts = batch
        print(f"  Batch images: {images.shape}")
        print(f"  Batch GTs: {gts.shape}")
        break
        
    print("Test batch:")
    for batch in test_loader:
        images, filenames = batch
        print(f"  Batch images: {images.shape}")
        print(f"  Batch filenames: {filenames}")
        break
    
    print("\nDataset loader tests completed successfully!")