# Brain Tumor MRI Classification with Explainability & Uncertainty Estimation

## 📋 Project Overview

This project develops a deep learning system for classifying brain MRI scans into four categories:
- Glioma
- Meningioma 
- Pituitary tumor
- No tumor

### Key Features
- **Transfer Learning**: Uses ResNet50/EfficientNetB0 for robust feature extraction
- **Explainability**: Grad-CAM visualizations show which brain regions influence predictions
- **Uncertainty Estimation**: Monte-Carlo Dropout flags low-confidence cases for human review

## 🏗️ Project Structure

```
brain_tumor_classification/
├── data/                   # Dataset storage
├── src/                    # Source code modules
├── models/                 # Trained model checkpoints
├── notebooks/              # Jupyter notebooks for experiments
├── results/                # Outputs, figures, and analysis
├── paper/                  # Research paper components
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   - Download the Brain Tumor MRI Dataset from Kaggle
   - Extract to `data/` directory

3. **Run Training**
   ```bash
   python src/train.py
   ```

## 📊 Success Criteria

- ≥85% classification accuracy on test set
- Generate Grad-CAM overlays highlighting tumor regions
- Demonstrate uncertainty-based referral improves accuracy
- Deliver reproducible research paper with figures and tables

## 🛠️ Tools & Technologies

- **Framework**: PyTorch
- **Libraries**: torchvision, scikit-learn, matplotlib, seaborn
- **Compute**: Compatible with Google Colab / Kaggle Notebooks (GPU)

## 📚 Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- 7,023 human brain MRI images
- 4 classes: glioma, meningioma, pituitary, no tumor
- High-quality medical imaging data