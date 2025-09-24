"""
Complete training script for Brain Tumor MRI Classification
This script can be run independently or used as a reference for the notebook
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_utils import create_data_splits, ImageTransforms, BrainMRIDataset, create_data_loaders
from models import BrainMRIClassifier, EarlyStopping, save_model, get_model_summary
from gradcam import GradCAM, visualize_gradcam_batch, save_gradcam_examples


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                early_stopping, num_epochs, device, save_path):
    """Complete training loop with validation and early stopping"""
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print()
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    save_model(model, save_path, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'final_epoch': epoch + 1
    })
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, data_loader, device, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("📊 Evaluating model...")
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return accuracy, all_predictions, all_labels, all_probabilities


def main():
    """Main training and evaluation pipeline"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Project paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    
    # Create directories
    for dir_path in [models_dir, results_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if dataset exists
    if not (data_dir / "Training").exists():
        print("❌ Dataset not found!")
        print("Please download the Brain Tumor MRI Dataset from:")
        print("https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        print(f"Extract to: {data_dir}")
        return
    
    # Create data splits
    print("📊 Creating data splits...")
    data_splits = create_data_splits(data_dir, val_split=0.2)
    
    if data_splits[0] is None:
        print("❌ Could not create data splits")
        return
    
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), (classes, class_to_idx, idx_to_class) = data_splits
    
    print(f"Classes: {classes}")
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Setup transforms
    transforms_manager = ImageTransforms(image_size=224)
    train_transforms = transforms_manager.get_train_transforms()
    val_transforms = transforms_manager.get_val_transforms()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
        train_transforms,
        val_transforms,
        batch_size=32,
        num_workers=4
    )
    
    # Create model
    print("🏗️ Creating model...")
    model = BrainMRIClassifier(
        num_classes=len(classes),
        model_name='resnet50',
        pretrained=True,
        dropout_rate=0.5
    )
    
    model = model.to(device)
    model.freeze_backbone()  # Start with frozen backbone
    
    # Print model summary
    summary = get_model_summary(model)
    print(f"Model: {summary['total_parameters']:,} parameters, {summary['model_size_mb']:.1f} MB")
    
    # Training configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    # Train model
    print("🎯 Starting training...")
    model_save_path = models_dir / "best_brain_tumor_model.pth"
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        num_epochs=50,
        device=device,
        save_path=model_save_path
    )
    
    # Evaluate on test set
    print("🔍 Evaluating on test set...")
    test_accuracy, predictions, labels, probabilities = evaluate_model(
        model, test_loader, device, classes
    )
    
    print(f"\n🎯 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Success criteria check
    if test_accuracy >= 0.85:
        print("✅ SUCCESS: Achieved target accuracy of 85%!")
    else:
        print("⚠️ Below target accuracy of 85%")
    
    # Generate Grad-CAM visualizations
    print("🎨 Generating Grad-CAM visualizations...")
    gradcam_dir = results_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)
    
    try:
        save_gradcam_examples(
            model, test_loader, classes, 'resnet50', device, 
            str(gradcam_dir), num_examples_per_class=2
        )
    except Exception as e:
        print(f"⚠️ Could not generate Grad-CAM: {e}")
    
    print("\n✅ Training and evaluation complete!")
    print(f"📁 Model saved: {model_save_path}")
    print(f"📁 Results saved: {results_dir}")


if __name__ == "__main__":
    main()