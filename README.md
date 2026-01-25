# Face Recognition using PCA-ANN

A comprehensive implementation of face recognition using Principal Component Analysis (PCA) for feature extraction and Artificial Neural Networks (ANN) for classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a face recognition system that combines two powerful techniques:
- **PCA (Principal Component Analysis)**: For dimensionality reduction and feature extraction (Eigenfaces)
- **ANN (Artificial Neural Network)**: For classification using Multi-Layer Perceptron (MLP)

The system can recognize faces from both Face and Iris biometric datasets with high accuracy.

## ✨ Features

- **Multiple Dataset Support**: Works with both face and iris recognition datasets
- **Eigenface Generation**: Extracts principal components to represent faces
- **Train-Test Split Evaluation**: 60-40 split for robust model evaluation
- **K-Value Analysis**: Evaluate performance with different numbers of principal components
- **Accuracy Visualization**: Generate plots showing accuracy vs. number of components
- **Imposter Detection**: Identify unknown faces not in the training set
- **Modular Design**: Clean, object-oriented implementation

## 📁 Project Structure

```
Face-Recognition-using-PCA-ANN/
│
├── pca_ann_face_recognition.py    # Main implementation file
├── Iris/                           # Iris dataset folder
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── person2/
│       └── ...
├── README.md                       # This file
└── accuracy_vs_k.png              # Generated accuracy plot
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/shoebbagwan/Face-Recognition-using-PCA-ANN.git
cd Face-Recognition-using-PCA-ANN
```

### Step 2: Install Required Packages
```bash
pip install numpy opencv-python scipy scikit-learn matplotlib
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import cv2; import numpy; import sklearn; print('All packages installed successfully!')"
```

## 📊 Dataset

### Dataset Structure
Organize your dataset in the following structure:

```
dataset/
├── faces/                  # Face dataset
│   ├── person1/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── person2/
│   │   ├── 1.jpg
│   │   └── ...
│   └── ...
│
└── Iris/                   # Iris dataset (included)
    ├── subject1/
    └── ...
```

### Dataset Requirements
- Images should be in common formats (JPG, PNG, BMP)
- Each person should have their own subdirectory
- Minimum 5-10 images per person recommended
- Images will be automatically resized to 92x112 pixels

### Popular Datasets You Can Use
- **AT&T (ORL) Face Database**: 40 subjects, 10 images each
- **Yale Face Database**: Various lighting conditions
- **LFW (Labeled Faces in the Wild)**: Real-world face images
- **Your own dataset**: Organize as shown above

## 💻 Usage

### Basic Usage

Run the main script:
```bash
python pca_ann_face_recognition.py
```

The program will prompt you to select a dataset:
```
Select dataset:
1. Face Dataset
2. Iris Dataset
Enter choice (1 or 2):
```

### Configuring Dataset Paths

Edit the paths in the `main()` function of `pca_ann_face_recognition.py`:

```python
# Update these paths to your dataset locations
face_dataset_path = r"C:\path\to\your\dataset\faces"
iris_dataset_path = r"C:\path\to\your\dataset\Iris"
```

### Advanced Usage

#### Train with Custom Parameters
```python
from pca_ann_face_recognition import PCA_ANN_FaceRecognition

model = PCA_ANN_FaceRecognition()
model.train(
    dataset_path="path/to/dataset",
    k=50,                           # Number of principal components
    hidden_layers=(100, 50),        # ANN architecture
    max_iter=500                    # Training iterations
)
```

#### Evaluate with Different K Values
```python
from pca_ann_face_recognition import evaluate_different_k_values

k_values = [10, 20, 30, 40, 50, 75, 100, 150]
evaluate_different_k_values("path/to/dataset", k_values)
```

#### Predict Single Image
```python
predicted_label, confidence = model.predict("path/to/test_image.jpg")
print(f"Predicted: {model.label_names[predicted_label]}")
print(f"Confidence: {confidence:.2f}")
```

## 🔬 How It Works

### 1. Data Loading
- Reads grayscale images from subdirectories
- Resizes all images to standard dimensions (92x112)
- Flattens each image into a column vector

### 2. PCA Pipeline

#### Step 1: Mean Face Calculation
```
mean_face = (1/N) * Σ(face_i)
```

#### Step 2: Mean Zero Alignment
```
φ = face_db - mean_face
```

#### Step 3: Covariance Matrix
```
C = φᵀ * φ
```

#### Step 4: Eigen Decomposition
- Compute eigenvalues and eigenvectors
- Sort by eigenvalues in descending order

#### Step 5: Feature Selection
- Select top k eigenvectors
- These form the "eigenfaces"

#### Step 6: Signature Generation
```
signatures = eigenfaces * φ
```

### 3. ANN Classification
- Multi-Layer Perceptron (MLP) classifier
- Default architecture: 2 hidden layers (100, 50 neurons)
- Trained on PCA-reduced features (signatures)
- Uses early stopping to prevent overfitting

### 4. Prediction
For a new face:
1. Subtract mean face
2. Project onto eigenfaces
3. Feed to trained ANN
4. Get predicted identity + confidence

## 📈 Results

### Performance Metrics

The system evaluates performance using:
- **Training Accuracy**: Model fit on training data
- **Test Accuracy**: Performance on unseen data (40% split)
- **Explained Variance**: How much information is retained by k components

### Example Output
```
============================================================
Evaluating with k=50 principal components
============================================================

Loading dataset...
Loaded 400 images from 40 persons
Image dimensions: 112 x 92

Training set: 240 images
Test set: 160 images

Explained variance with 50 components: 89.34%

Training ANN...
Test Accuracy: 94.38%
```

### Accuracy vs K Components

The generated plot (`accuracy_vs_k.png`) shows:
- How accuracy changes with different numbers of principal components
- Helps identify optimal k value
- Typically shows accuracy increasing then plateauing

## 🛠️ Algorithm Details

### PCA (Principal Component Analysis)
- **Purpose**: Dimensionality reduction
- **Input**: High-dimensional face vectors (10,304 dimensions for 92x112 images)
- **Output**: Low-dimensional signatures (k dimensions, typically 50-100)
- **Benefit**: Reduces computation, removes noise, extracts key features

### Eigenfaces
- Eigenvectors of the covariance matrix
- Represent "principal components" of faces
- Each eigenface captures a specific facial variation
- Linear combination can reconstruct any face

### ANN (Artificial Neural Network)
- **Architecture**: Multi-Layer Perceptron (MLP)
- **Input Layer**: k features (from PCA)
- **Hidden Layers**: (100, 50) neurons with ReLU activation
- **Output Layer**: N neurons (one per person)
- **Optimizer**: Adam
- **Loss**: Cross-entropy

## 📦 Requirements

```
numpy>=1.19.0
opencv-python>=4.5.0
scipy>=1.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

Create a `requirements.txt` file with the above content for easy installation.

## ⚙️ Configuration Options

### Adjustable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 50 | Number of principal components |
| `hidden_layers` | (100, 50) | ANN hidden layer sizes |
| `max_iter` | 500 | Maximum training iterations |
| `test_size` | 0.4 | Test set proportion (40%) |
| `img_size` | (92, 112) | Standard image dimensions |

### Tuning Tips

- **Increase k**: Better accuracy but slower computation
- **Decrease k**: Faster but may lose important features
- **More hidden neurons**: Can learn complex patterns but risk overfitting
- **More training images**: Generally improves accuracy

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'cv2'`
```bash
# Solution:
pip install opencv-python
```

**Issue**: `ERROR: Dataset path does not exist`
```python
# Solution: Check your path format
# Windows:
dataset_path = r"C:\Users\...\dataset\faces"
# Linux/Mac:
dataset_path = "/home/user/dataset/faces"
```

**Issue**: Low accuracy
- Try increasing k (number of components)
- Ensure sufficient images per person (minimum 5-10)
- Check image quality and consistency
- Verify dataset organization

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Shoeb Bagwan**
- GitHub: [@shoebbagwan](https://github.com/shoebbagwan)

## 🙏 Acknowledgments

- Based on the Eigenfaces algorithm by Turk and Pentland (1991)
- Inspired by various PCA face recognition implementations
- Thanks to the open-source community for libraries and datasets

## 📚 References

1. Turk, M., & Pentland, A. (1991). "Eigenfaces for Recognition". Journal of Cognitive Neuroscience.
2. Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). "Eigenfaces vs. Fisherfaces".
3. Scikit-learn Documentation: https://scikit-learn.org/
4. OpenCV Documentation: https://docs.opencv.org/

**Happy Face Recognition! **
