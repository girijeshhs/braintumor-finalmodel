"""
Model architectures and utilities for Brain MRI Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional


class BrainMRIClassifier(nn.Module):
    """
    Transfer learning model for brain MRI classification
    Supports ResNet50 and EfficientNet architectures
    """
    
    def __init__(self, num_classes: int = 4, model_name: str = 'resnet50', 
                 pretrained: bool = True, dropout_rate: float = 0.5):
        super(BrainMRIClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        # Load pretrained backbone
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # Remove final layer
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Enable dropout during inference for MC Dropout
        self.mc_dropout = False
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply classifier
        if self.mc_dropout:
            # Enable dropout during inference
            self.classifier.train()
        
        logits = self.classifier(features)
        return logits
    
    def enable_mc_dropout(self):
        """Enable Monte Carlo Dropout for uncertainty estimation"""
        self.mc_dropout = True
        
    def disable_mc_dropout(self):
        """Disable Monte Carlo Dropout for normal inference"""
        self.mc_dropout = False
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(num_classes: int = 4, model_name: str = 'resnet50', 
                pretrained: bool = True, dropout_rate: float = 0.5) -> BrainMRIClassifier:
    """
    Create a brain MRI classification model
    
    Args:
        num_classes: Number of output classes
        model_name: Name of the backbone model ('resnet50' or 'efficientnet_b0')
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization
    
    Returns:
        BrainMRIClassifier model
    """
    model = BrainMRIClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    return model


def get_model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)) -> Dict:
    """
    Get a summary of model parameters and memory usage
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    
    Returns:
        Dictionary with model summary information
    """
    device = next(model.parameters()).device
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage
    dummy_input = torch.randn(1, *input_size).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'input_size': input_size,
        'output_size': output.shape,
        'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
    }
    
    return summary


class EarlyStopping:
    """Early stopping callback to prevent overfitting"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if early stopping criteria is met
        
        Args:
            val_loss: Current validation loss
            model: PyTorch model
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


def save_model(model: nn.Module, filepath: str, 
              model_info: Optional[Dict] = None):
    """
    Save model checkpoint with additional information
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        model_info: Additional information to save with the model
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_info': model_info or {}
    }
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to: {filepath}")


def load_model(model: nn.Module, filepath: str) -> Dict:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model to load weights into
        filepath: Path to the saved model
    
    Returns:
        Dictionary with model information
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from: {filepath}")
    return checkpoint.get('model_info', {})