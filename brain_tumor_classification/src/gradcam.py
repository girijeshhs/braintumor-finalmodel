"""
Grad-CAM implementation for explainable AI in brain tumor classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, use predicted class)
        
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Calculate importance weights
        weights = torch.mean(gradients, dim=(1, 2))  # Global average pooling
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / torch.max(cam)
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, input_image: np.ndarray, cam: np.ndarray, 
                     alpha: float = 0.4) -> np.ndarray:
        """
        Overlay Grad-CAM on original image
        
        Args:
            input_image: Original image (H, W, C) in range [0, 255]
            cam: Grad-CAM heatmap (H, W)
            alpha: Transparency factor for overlay
        
        Returns:
            Visualization image with CAM overlay
        """
        # Resize CAM to match input image
        cam_resized = cv2.resize(cam, (input_image.shape[1], input_image.shape[0]))
        
        # Normalize CAM to [0, 255]
        cam_normalized = np.uint8(255 * cam_resized)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlayed = input_image * alpha + heatmap * (1 - alpha)
        overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
        
        return overlayed
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_target_layer_name(model: nn.Module, model_name: str) -> str:
    """
    Get the appropriate target layer name for Grad-CAM based on model architecture
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
    
    Returns:
        Target layer name for Grad-CAM
    """
    if model_name == 'resnet50':
        return 'backbone.layer4'  # Last convolutional layer before global pooling
    elif model_name == 'efficientnet_b0':
        return 'backbone.features'  # Last feature extraction layer
    else:
        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def visualize_gradcam_batch(model: nn.Module, data_loader, class_names: List[str],
                           model_name: str, device: torch.device, num_samples: int = 8):
    """
    Visualize Grad-CAM for a batch of images
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing images
        class_names: List of class names
        model_name: Name of the model architecture
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get target layer for Grad-CAM
    target_layer = get_target_layer_name(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    
    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # Setup subplot
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
    fig.suptitle('Grad-CAM Visualizations', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # Original image
        original_img = images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize image (reverse ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = original_img * std + mean
        original_img = np.clip(original_img * 255, 0, 255).astype(np.uint8)
        
        # Generate Grad-CAM
        input_tensor = images[i:i+1]  # Keep batch dimension
        cam = grad_cam.generate_cam(input_tensor)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1).max().item()
        
        # Visualize
        axes[0, i].imshow(original_img)
        axes[0, i].set_title(f'Original\\nTrue: {class_names[labels[i]]}', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(cam, cmap='jet')
        axes[1, i].set_title(f'Grad-CAM Heatmap', fontsize=10)
        axes[1, i].axis('off')
        
        # Overlay
        overlayed = grad_cam.visualize_cam(original_img, cam)
        axes[2, i].imshow(overlayed)
        axes[2, i].set_title(f'Overlay\\nPred: {class_names[predicted_class]}\\nConf: {confidence:.3f}', fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Clean up
    grad_cam.remove_hooks()


def save_gradcam_examples(model: nn.Module, data_loader, class_names: List[str],
                         model_name: str, device: torch.device, save_dir: str,
                         num_examples_per_class: int = 2):
    """
    Save Grad-CAM examples for each class
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing images
        class_names: List of class names
        model_name: Name of the model architecture
        device: Device to run inference on
        save_dir: Directory to save examples
        num_examples_per_class: Number of examples to save per class
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    target_layer = get_target_layer_name(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    
    class_counts = {class_name: 0 for class_name in class_names}
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            
            for i in range(images.size(0)):
                label = labels[i].item()
                class_name = class_names[label]
                
                if class_counts[class_name] >= num_examples_per_class:
                    continue
                
                # Generate Grad-CAM
                input_tensor = images[i:i+1]
                cam = grad_cam.generate_cam(input_tensor)
                
                # Prepare original image
                original_img = images[i].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                original_img = original_img * std + mean
                original_img = np.clip(original_img * 255, 0, 255).astype(np.uint8)
                
                # Create overlay
                overlayed = grad_cam.visualize_cam(original_img, cam)
                
                # Save figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(original_img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(cam, cmap='jet')
                axes[1].set_title('Grad-CAM Heatmap')
                axes[1].axis('off')
                
                axes[2].imshow(overlayed)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
                
                plt.suptitle(f'Class: {class_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                filename = f"{class_name}_{class_counts[class_name]+1}.png"
                plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
                plt.close()
                
                class_counts[class_name] += 1
            
            # Check if we have enough examples for all classes
            if all(count >= num_examples_per_class for count in class_counts.values()):
                break
    
    grad_cam.remove_hooks()
    print(f"Grad-CAM examples saved to: {save_dir}")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} examples")