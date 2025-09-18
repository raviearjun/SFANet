#!/usr/bin/env python3
"""
End-to-end testing script for SFANet competition pipeline.
Tests all components from preprocessing to inference.
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n[STEP {step}] {description}")
    print("-" * 50)

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists

def run_command(command, description, check_output=True):
    """Run a command and handle errors"""
    print(f"\nRunning: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=check_output, 
                              text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✓ Success")
            if check_output and result.stdout:
                print("Output:")
                print(result.stdout[:500])  # Limit output length
            return True
        else:
            print("✗ Failed")
            if check_output and result.stderr:
                print("Error:")
                print(result.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout")
        return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def test_prerequisites():
    """Test system prerequisites"""
    print_step(1, "Testing Prerequisites")
    
    # Test Python imports
    imports_to_test = [
        'torch',
        'torchvision', 
        'h5py',
        'scipy',
        'PIL',
        'cv2',
        'numpy',
        'pandas'
    ]
    
    print("Checking Python imports:")
    all_imports_ok = True
    
    for module in imports_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - MISSING")
            all_imports_ok = False
    
    # Test CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        print(f"\nCUDA Status:")
        print(f"  ✓ Available: {cuda_available}")
        print(f"  ✓ GPU Count: {gpu_count}")
        if cuda_available:
            print(f"  ✓ Current GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("  ✗ Could not check CUDA status")
        cuda_available = False
    
    return all_imports_ok and cuda_available

def test_dataset_structure():
    """Test dataset structure"""
    print_step(2, "Testing Dataset Structure")
    
    dataset_path = "./dataset"
    required_paths = [
        ("Dataset root", dataset_path),
        ("Training images", f"{dataset_path}/train/images"),
        ("Training labels", f"{dataset_path}/train/labels"),
        ("Test images", f"{dataset_path}/test/images"),
        ("Sample submission", f"{dataset_path}/sample_submission.csv")
    ]
    
    all_paths_ok = True
    for desc, path in required_paths:
        if not check_file_exists(path, desc):
            all_paths_ok = False
    
    # Count files
    if all_paths_ok:
        try:
            train_images = len([f for f in os.listdir(f"{dataset_path}/train/images") 
                              if f.endswith('.jpg')])
            train_labels = len([f for f in os.listdir(f"{dataset_path}/train/labels") 
                              if f.endswith('.json')])
            test_images = len([f for f in os.listdir(f"{dataset_path}/test/images") 
                             if f.endswith('.jpg')])
            
            print(f"\nFile counts:")
            print(f"  Training images: {train_images}")
            print(f"  Training labels: {train_labels}")
            print(f"  Test images: {test_images}")
            
            # Check correspondence
            if train_images == train_labels:
                print("  ✓ Training images and labels match")
            else:
                print("  ✗ Training images and labels don't match")
                all_paths_ok = False
                
        except Exception as e:
            print(f"  ✗ Error counting files: {e}")
            all_paths_ok = False
    
    return all_paths_ok

def test_preprocessing():
    """Test density map generation"""
    print_step(3, "Testing Density Map Generation")
    
    # Run preprocessing (limit to small subset for testing)
    success = run_command(
        "python density_map_competition.py",
        "Generate density maps from JSON annotations"
    )
    
    if success:
        # Check output directory
        output_dir = "./dataset/train/new_data"
        if os.path.exists(output_dir):
            h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]
            print(f"  ✓ Generated {len(h5_files)} density map files")
            
            # Test loading one file
            if h5_files:
                try:
                    import h5py
                    sample_file = os.path.join(output_dir, h5_files[0])
                    with h5py.File(sample_file, 'r') as f:
                        density_shape = f['density'].shape
                        attention_shape = f['attention'].shape
                        gt_count = f['gt'][()]
                    
                    print(f"  ✓ Sample density map shape: {density_shape}")
                    print(f"  ✓ Sample attention map shape: {attention_shape}")
                    print(f"  ✓ Sample ground truth count: {gt_count}")
                    
                except Exception as e:
                    print(f"  ✗ Error loading sample .h5 file: {e}")
                    success = False
        else:
            print("  ✗ Output directory not created")
            success = False
    
    return success

def test_dataset_loader():
    """Test dataset loader"""
    print_step(4, "Testing Dataset Loader")
    
    success = run_command(
        "python -c \"from dataset_competition import DatasetCompetition; print('Dataset loader imported successfully')\"",
        "Import dataset loader"
    )
    
    if success:
        # Test dataset creation
        test_script = """
import sys
sys.path.append('.')
from dataset_competition import DatasetCompetition

try:
    # Test training dataset
    train_ds = DatasetCompetition('./dataset', is_train=True, is_test=False)
    print(f'Training dataset: {len(train_ds)} samples')
    
    # Test validation dataset  
    val_ds = DatasetCompetition('./dataset', is_train=False, is_test=False)
    print(f'Validation dataset: {len(val_ds)} samples')
    
    # Test test dataset
    test_ds = DatasetCompetition('./dataset', is_train=False, is_test=True)
    print(f'Test dataset: {len(test_ds)} samples')
    
    print('Dataset loader test: SUCCESS')
    
except Exception as e:
    print(f'Dataset loader test: FAILED - {e}')
    sys.exit(1)
"""
        
        success = run_command(
            f"python -c \"{test_script}\"",
            "Test dataset loader functionality"
        )
    
    return success

def test_model_creation():
    """Test model creation and basic forward pass"""
    print_step(5, "Testing Model Creation")
    
    test_script = """
import torch
from models import Model

try:
    # Create model
    model = Model()
    print(f'Model created successfully')
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 400, 400)
    
    with torch.no_grad():
        density_out, attention_out = model(dummy_input)
    
    print(f'Forward pass successful')
    print(f'Density output shape: {density_out.shape}')
    print(f'Attention output shape: {attention_out.shape}')
    
    print('Model test: SUCCESS')
    
except Exception as e:
    print(f'Model test: FAILED - {e}')
    import sys
    sys.exit(1)
"""
    
    success = run_command(
        f"python -c \"{test_script}\"",
        "Test model creation and forward pass"
    )
    
    return success

def test_training_setup():
    """Test training setup (without actually training)"""
    print_step(6, "Testing Training Setup")
    
    # Create checkpoint directory
    checkpoint_dir = "./checkpoints_test"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Test training script import and basic setup
    success = run_command(
        f"python train_competition.py --epoch 1 --bs 2 --save_path {checkpoint_dir} --data_path ./dataset",
        "Test training script (1 epoch, small batch)",
        check_output=False  # Training output can be very long
    )
    
    if success:
        # Check if checkpoint was created
        latest_checkpoint = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
        if check_file_exists(latest_checkpoint, "Latest checkpoint"):
            print("  ✓ Training test completed successfully")
        else:
            print("  ✗ Training checkpoint not created")
            success = False
    
    # Cleanup test checkpoint
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    return success

def test_inference_setup():
    """Test inference setup"""
    print_step(7, "Testing Inference Setup")
    
    # Since we don't have a real trained model, test the inference script setup
    success = run_command(
        "python -c \"from inference_competition import main; print('Inference script imported successfully')\"",
        "Import inference script"
    )
    
    return success

def generate_summary_report(results):
    """Generate final summary report"""
    print_section("TEST SUMMARY REPORT")
    
    test_names = [
        "Prerequisites Check",
        "Dataset Structure", 
        "Density Map Generation",
        "Dataset Loader",
        "Model Creation",
        "Training Setup", 
        "Inference Setup"
    ]
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    print("\nDetailed Results:")
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {i+1}. {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The competition pipeline is ready.")
        print("\nNext steps:")
        print("1. Run full training: python train_competition.py")
        print("2. Generate predictions: python inference_competition.py")
        print("3. Submit your results!")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")
        print("\nTroubleshooting:")
        print("- Check README_COMPETITION.md for setup instructions")
        print("- Ensure all dependencies are installed")
        print("- Verify dataset structure matches requirements")
        return False

def main():
    """Run all tests"""
    print_section("SFANet Competition Pipeline Testing")
    print("This script will test the complete pipeline from preprocessing to inference.")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    tests = [
        test_prerequisites,
        test_dataset_structure,
        test_preprocessing,
        test_dataset_loader,
        test_model_creation,
        test_training_setup,
        test_inference_setup
    ]
    
    results = []
    start_time = time.time()
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            
            if not result:
                print(f"\n⚠️  Test failed: {test_func.__name__}")
                print("You may want to fix this before continuing...")
                
        except Exception as e:
            print(f"\n💥 Test crashed: {test_func.__name__}")
            print(f"Error: {e}")
            results.append(False)
    
    total_time = time.time() - start_time
    
    # Generate summary
    success = generate_summary_report(results)
    
    print(f"\nTotal testing time: {total_time:.1f}s")
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)