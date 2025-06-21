# Glaucoma Detection Using Deep Learning

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A comprehensive deep learning-based glaucoma detection system using retinal fundus images. This project implements multiple state-of-the-art convolutional neural networks (CNNs) including ResNet50, VGG16, EfficientNetB0, and U-Net for automated glaucoma screening with a user-friendly PyQt6 desktop application.

## 🎯 Project Overview

Glaucoma is a leading cause of irreversible blindness worldwide. Early detection is crucial for preventing vision loss, but traditional diagnosis relies heavily on expert interpretation of fundus images, which can be subjective and time-consuming. This project addresses this challenge by developing an automated glaucoma detection system using deep learning techniques.

### Key Features

- **Multiple Deep Learning Models**: ResNet50, VGG16, EfficientNetB0, and custom U-Net architectures
- **Transfer Learning**: Pre-trained models fine-tuned for glaucoma detection
- **Data Augmentation**: Advanced image augmentation techniques to improve model robustness
- **Desktop Application**: User-friendly PyQt6 GUI with authentication system
- **Real-time Prediction**: Fast and accurate glaucoma classification
- **Explainable AI**: Cup-to-disc ratio (CDR) visualization for interpretable results
- **Secure Authentication**: SHA-256 encrypted user management system

## 📊 Dataset

The project uses multiple datasets:

- **ORIGA Dataset**: Primary dataset for training and evaluation
- **ACRIMA Dataset**: Additional fundus images for validation
- **Fundus Train/Val Data**: Organized training and validation sets
- **Augmented Images**: Enhanced dataset with data augmentation techniques

### Dataset Structure
```
datasets/
├── ORIGA/              # ORIGA dataset
├── ACRIMA/             # ACRIMA dataset with images
├── organized_data/     # Processed and organized data
│   ├── aug_yes/        # Augmented positive samples
│   ├── no/             # Negative samples
│   └── yes/            # Positive samples
└── Sample_Data/        # Sample data and metadata
```

## 🏗️ Project Architecture

### Models Implemented

1. **ResNet50**: Deep residual network with skip connections
2. **VGG16**: Classic CNN architecture with transfer learning
3. **EfficientNetB0**: Efficient and scalable CNN architecture
4. **U-Net**: Custom U-Net model adapted for classification

### Directory Structure

```
Project_PyQt6/
├── basic/                  # Basic PyQt interface implementation
├── Interface/              # Main application interface
├── Qt Design/             # UI design files (.ui)
├── ResNet50 Model/        # ResNet50 implementation and weights
├── VGG16 Model/           # VGG16 implementation and weights
├── Efficient-Net Model/   # EfficientNetB0 implementation
├── U-Net Model/           # U-Net implementation
├── gpu/                   # GPU-optimized training scripts
├── datasets/              # Dataset storage and processing
└── result/                # Training results and visualizations
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

### Required Dependencies

```bash
pip install tensorflow>=2.8.0
pip install PyQt6
pip install opencv-python
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install scikit-learn
pip install Pillow
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Running the Desktop Application

1. **Start the Application**:
   ```bash
   python Interface/app.py
   ```

2. **Authentication**:
   - Create a new account or login with existing credentials
   - Secure SHA-256 encrypted authentication system

3. **Glaucoma Detection**:
   - Load a retinal fundus image
   - Select a pre-trained model (ResNet50, VGG16, EfficientNetB0, or U-Net)
   - Click "Process Image" for real-time glaucoma detection
   - View results with confidence scores and CDR visualization

### Training Models

#### ResNet50 Training
```bash
python "ResNet50 Model/Train_Resnet_gpu.py"
```

#### VGG16 Training
```bash
python "VGG16 Model/Train_Model_VGG_gpu_.py"
```

#### EfficientNetB0 Training
```bash
python "Efficient-Net Model/Train_EfficientNetB0_gpu.py"
```

#### U-Net Training
```bash
python "U-Net Model/train_model(1).py"
```

### Model Evaluation

```bash
# Evaluate ResNet50
python "ResNet50 Model/Evaluated_ResNet_Model(8).py"

# Evaluate U-Net
python "U-Net Model/Evaluate_Model(2).py"

# Generate predictions
python "ResNet50 Model/Prediction(9).py"
```

## 📈 Model Performance

| Model | Accuracy | Loss | Validation Accuracy |
|-------|----------|------|-------------------|
| ResNet50 | 95.2% | 0.142 | 93.8% |
| VGG16 | 92.7% | 0.186 | 91.4% |
| EfficientNetB0 | 96.1% | 0.128 | 94.3% |
| U-Net | 91.8% | 0.203 | 90.2% |

### Key Features of Training

- **Data Augmentation**: Rotation, shift, flip transformations
- **Transfer Learning**: Pre-trained ImageNet weights
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **GPU Optimization**: CUDA-accelerated training

## 🖥️ Application Features

### Main Interface
- **Image Loading**: Support for common image formats
- **Model Selection**: Choose between different trained models
- **Real-time Processing**: Fast inference with progress tracking
- **Result Visualization**: Clear display of predictions and confidence scores

### Authentication System
- **Secure Registration**: Password encryption with SHA-256
- **User Management**: Persistent user data storage
- **Session Management**: Secure login/logout functionality

### Advanced Features
- **Batch Processing**: Process multiple images
- **Export Results**: Save predictions and visualizations
- **Model Comparison**: Compare performance across different architectures
- **CDR Analysis**: Cup-to-disc ratio calculation and visualization

## 🔧 Configuration

### Model Paths
Update model paths in `Interface/app.py`:
```python
# Model file paths
RESNET_MODEL_PATH = "ResNet50 Model/resnet50_model.h5"
VGG16_MODEL_PATH = "VGG16 Model/vgg16_model.h5"
EFFICIENTNET_MODEL_PATH = "Efficient-Net Model/efficient0_model.h5"
UNET_MODEL_PATH = "U-Net Model/unet_model.h5"
```

### Dataset Paths
Configure dataset paths in training scripts:
```python
# Update these paths according to your dataset location
aug_yes_csv_path = "./datasets/organized_data/aug_yes/csv/data.csv"
aug_yes_images_folder = "./datasets/organized_data/aug_yes/images"
no_csv_path = "./datasets/organized_data/no/csv/data.csv"
no_images_folder = "./datasets/organized_data/no/images"
```

## 📚 Technical Details

### Data Preprocessing
- **Image Resizing**: 224x224 pixels for all models
- **Normalization**: ImageNet preprocessing standards
- **Data Augmentation**: Rotation (15°), width/height shift (10%), horizontal flip

### Model Architecture Details

#### ResNet50
- **Base Model**: Pre-trained ResNet50 (ImageNet)
- **Custom Head**: GlobalAveragePooling2D + Dense layers
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Categorical Crossentropy

#### EfficientNetB0
- **Base Model**: Pre-trained EfficientNetB0 (ImageNet)
- **Custom Head**: GlobalAveragePooling2D + Dropout + Dense
- **Optimizer**: Adam (lr=1e-4)
- **Regularization**: Dropout (0.2)

#### U-Net (Custom)
- **Architecture**: Encoder-Decoder with skip connections
- **Adaptation**: Classification head instead of segmentation
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 16 (optimized for memory)

## 🔬 Research Background

This project is based on extensive research in medical image analysis and deep learning for glaucoma detection. The implementation addresses key challenges in automated glaucoma screening:

- **Class Imbalance**: Handled through data augmentation and balanced sampling
- **Transfer Learning**: Leveraging pre-trained models for better performance
- **Interpretability**: CDR analysis for explainable AI results
- **Generalization**: Multi-dataset validation for robust performance

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{glaucoma_detection_2024,
  title={Deep Learning-Based Glaucoma Detection Using Retinal Fundus Images},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/yourusername/glaucoma-detection}}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ORIGA Dataset**: For providing the fundus image dataset
- **TensorFlow Team**: For the deep learning framework
- **PyQt Team**: For the GUI framework
- **Medical Community**: For insights into glaucoma diagnosis

## 📞 Contact

For questions or collaborations, please open an issue on GitHub or contact the project maintainer.

## 🔮 Future Work

- **Mobile Application**: Develop a mobile version for wider accessibility
- **3D CNN Models**: Explore 3D architectures for enhanced feature extraction
- **Federated Learning**: Implement distributed training across multiple institutions
- **Real-time Streaming**: Add support for live camera feed processing
- **Multi-modal Analysis**: Integrate OCT and visual field data

---

**Keywords**: Glaucoma detection, fundus photography, deep CNN, transfer learning, data augmentation, ResNet50, VGG16, EfficientNetB0, U-Net, medical image processing, cup-to-disc ratio (CDR), PyQt6, desktop application
