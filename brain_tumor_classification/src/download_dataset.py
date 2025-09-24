"""
Script to download and prepare the Brain Tumor MRI Dataset from Kaggle
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
import shutil

def download_dataset_instructions():
    """Print instructions for downloading the dataset"""
    
    print("📥 Brain Tumor MRI Dataset Download Instructions")
    print("=" * 60)
    print()
    print("🔗 Dataset URL: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    print()
    print("📋 Manual Download Steps:")
    print("1. Go to the Kaggle dataset page")
    print("2. Click 'Download' button")
    print("3. Extract the downloaded ZIP file")
    print("4. Move the contents to this project's data/ directory")
    print()
    print("📁 Expected directory structure after extraction:")
    print("data/")
    print("├── Training/")
    print("│   ├── glioma/")
    print("│   ├── meningioma/")
    print("│   ├── notumor/")
    print("│   └── pituitary/")
    print("└── Testing/")
    print("    ├── glioma/")
    print("    ├── meningioma/")
    print("    ├── notumor/")
    print("    └── pituitary/")
    print()
    print("🚀 Alternative: Use Kaggle CLI (if you have API credentials)")
    print("1. Install Kaggle CLI: pip install kaggle")
    print("2. Set up API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
    print("3. Run: kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset")
    print("4. Extract and organize the files as shown above")
    print()

def verify_dataset_structure(data_dir):
    """Verify the dataset has the correct structure"""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return False, f"Data directory not found: {data_path}"
    
    training_path = data_path / "Training"
    testing_path = data_path / "Testing"
    
    if not training_path.exists():
        return False, f"Training directory not found: {training_path}"
    
    if not testing_path.exists():
        return False, f"Testing directory not found: {testing_path}"
    
    expected_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Check training classes
    for class_name in expected_classes:
        class_path = training_path / class_name
        if not class_path.exists():
            return False, f"Training class directory not found: {class_path}"
        
        # Count images
        image_count = len(list(class_path.glob("*.jpg")))
        if image_count == 0:
            return False, f"No images found in: {class_path}"
    
    # Check testing classes
    for class_name in expected_classes:
        class_path = testing_path / class_name
        if not class_path.exists():
            return False, f"Testing class directory not found: {class_path}"
        
        # Count images
        image_count = len(list(class_path.glob("*.jpg")))
        if image_count == 0:
            return False, f"No images found in: {class_path}"
    
    return True, "Dataset structure is correct!"

def main():
    """Main function to handle dataset download and verification"""
    
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    print("🧠 Brain Tumor MRI Dataset Setup")
    print("=" * 40)
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print()
    
    # Check if dataset already exists
    is_valid, message = verify_dataset_structure(data_dir)
    
    if is_valid:
        print("✅ Dataset is already properly set up!")
        print(message)
        
        # Count total images
        training_path = data_dir / "Training"
        testing_path = data_dir / "Testing"
        
        total_train = 0
        total_test = 0
        
        for class_name in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            train_count = len(list((training_path / class_name).glob("*.jpg")))
            test_count = len(list((testing_path / class_name).glob("*.jpg")))
            
            total_train += train_count
            total_test += test_count
            
            print(f"{class_name:12s}: Train={train_count:4d}, Test={test_count:3d}")
        
        print(f"{'='*40}")
        print(f"{'Total':12s}: Train={total_train:4d}, Test={total_test:3d}")
        
    else:
        print("❌ Dataset not found or incomplete!")
        print(f"Issue: {message}")
        print()
        download_dataset_instructions()

if __name__ == "__main__":
    main()