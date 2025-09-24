"""
Data utilities for Brain MRI Classification
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BrainMRIDataset(Dataset):
    """Custom Dataset for Brain MRI Classification"""
    
    def __init__(self, image_paths: List[Path], labels: List[int], 
                 transforms: Optional[A.Compose] = None, class_to_idx: Optional[Dict] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.class_to_idx = class_to_idx or {}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        return image, label


class ImageTransforms:
    """Class to handle image transformations for training and validation"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        
        # ImageNet statistics for normalization (required for transfer learning)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def get_train_transforms(self) -> A.Compose:
        """Get augmented transforms for training data"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ElasticTransform(p=0.3, alpha=1, sigma=50, alpha_affine=50),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def get_val_transforms(self) -> A.Compose:
        """Get transforms for validation/test data (no augmentation)"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def get_grad_cam_transforms(self) -> A.Compose:
        """Get transforms for Grad-CAM (no normalization for visualization)"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            ToTensorV2()
        ])


def create_data_splits(data_dir: Path, val_split: float = 0.2) -> Tuple:
    """
    Create train/validation/test splits from the dataset
    
    Args:
        data_dir: Path to data directory
        val_split: Fraction of training data to use for validation
    
    Returns:
        Tuple containing data splits and class information
    """
    
    if not data_dir.exists():
        print(f"⚠️ Data directory not found: {data_dir}")
        return None, None, None, None
    
    training_path = data_dir / "Training"
    testing_path = data_dir / "Testing"
    
    if not training_path.exists():
        print(f"⚠️ Training directory not found: {training_path}")
        return None, None, None, None
    
    # Get all classes
    classes = sorted([d.name for d in training_path.iterdir() if d.is_dir()])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    
    # Collect all training images
    all_image_paths = []
    all_labels = []
    
    for class_name in classes:
        class_path = training_path / class_name
        class_images = list(class_path.glob("*.jpg"))
        
        all_image_paths.extend(class_images)
        all_labels.extend([class_to_idx[class_name]] * len(class_images))
    
    # Split training data into train/validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, all_labels, 
        test_size=val_split, 
        stratify=all_labels, 
        random_state=42
    )
    
    # Collect test images if available
    test_paths = []
    test_labels = []
    
    if testing_path.exists():
        for class_name in classes:
            test_class_path = testing_path / class_name
            if test_class_path.exists():
                test_class_images = list(test_class_path.glob("*.jpg"))
                test_paths.extend(test_class_images)
                test_labels.extend([class_to_idx[class_name]] * len(test_class_images))
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), (classes, class_to_idx, idx_to_class)


def create_data_loaders(train_data: Tuple, val_data: Tuple, test_data: Tuple,
                       train_transforms: A.Compose, val_transforms: A.Compose,
                       batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing
    
    Args:
        train_data: Tuple of (image_paths, labels) for training
        val_data: Tuple of (image_paths, labels) for validation
        test_data: Tuple of (image_paths, labels) for testing
        train_transforms: Albumentations transforms for training
        val_transforms: Albumentations transforms for validation/testing
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    # Create datasets
    train_dataset = BrainMRIDataset(train_paths, train_labels, train_transforms)
    val_dataset = BrainMRIDataset(val_paths, val_labels, val_transforms)
    test_dataset = BrainMRIDataset(test_paths, test_labels, val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader