# Fashion MNIST Image Classification

A deep learning project that implements Convolutional Neural Networks (CNN) to classify fashion items from the Fashion MNIST dataset using TensorFlow and Keras.

## 📋 Project Overview

This project demonstrates image classification using neural networks on the Fashion MNIST dataset, which contains 70,000 grayscale images of 10 different fashion categories. The implementation includes data preprocessing, model building, training, and evaluation using CNN architectures.

## 🎯 Dataset

**Fashion MNIST** is a dataset of Zalando's article images consisting of:
- **Training set**: 60,000 images (28x28 pixels)
- **Test set**: 10,000 images (28x28 pixels)
- **Classes**: 10 fashion categories

### Class Labels
```
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow 2.x**
- **Keras**
- **NumPy**
- **Matplotlib**

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/khadija199904/Classification_Images_avec_R-seau_Neuronal.git
cd Classification_Images_avec_R-seau_Neuronal
```

2. Install required packages:
```bash
pip install tensorflow keras numpy matplotlib
```

## 🚀 Usage

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook Classification_fashon_Mnist.ipynb
```

2. Or run it directly in Google Colab:
   - Click the "Open in Colab" badge at the top of the notebook
   - Run all cells sequentially

### Basic Implementation

```python
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0
```

## 🏗️ Model Architecture

The project implements a Convolutional Neural Network with the following architecture:

```
Model: "first_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 32, 32, 32)        896       
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
max_pooling2d (MaxPooling2D) (None, 8, 8, 32)          0         
conv2d (Conv2D)             (None, 6, 6, 64)          18,496    
max_pooling2d (MaxPooling2D) (None, 3, 3, 64)          0         
flatten (Flatten)           (None, 576)               0         
dense (Dense)               (None, 64)                36,928    
Output (Dense)              (None, 10)                650       
=================================================================
Total params: 56,970 (222.54 KB)
Trainable params: 56,970 (222.54 KB)
Non-trainable params: 0 (0.00 B)
```

### Key Features:
- **2 Convolutional Layers**: For feature extraction
- **3 Max Pooling Layers**: For dimensionality reduction
- **2 Dense Layers**: For classification
- **Activation Functions**: ReLU for hidden layers, Softmax for output layer

## 📊 Data Preprocessing

- Image dimensions verified: 60,000 training images and 10,000 test images (28x28 pixels)
- Pixel values normalized to [0, 1] range
- Labels checked for 10 unique classes (0-9)
- Data visualization to understand the dataset structure

## 🔍 Key Features

- ✅ Data loading and preprocessing
- ✅ Data visualization with sample images
- ✅ CNN model architecture implementation
- ✅ Model training and evaluation
- ✅ Performance metrics analysis
- ✅ Support for Google Colab

## 📈 Results

The model achieves classification across 10 fashion categories using a compact CNN architecture with approximately 57K parameters.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Khadija**
- GitHub: [@khadija199904](https://github.com/khadija199904)

## 🙏 Acknowledgments

- Fashion MNIST dataset provided by Zalando Research
- TensorFlow and Keras teams for the excellent deep learning frameworks
- Google Colab for providing free GPU resources

## 📚 References

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)

---

**Note**: Make sure you have sufficient computational resources (preferably GPU) for faster training. Google Colab provides free GPU access which is recommended for this project.
