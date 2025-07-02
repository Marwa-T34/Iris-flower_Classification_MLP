# 🌸 MLP-Based Iris Flower Classification

## Overview  
This project implements a simple **Multi-Layer Perceptron (MLP)** neural network to classify iris flowers into one of three species:

- **Setosa**  
- **Versicolor**  
- **Virginica**

## Dataset  
  
The Iris dataset is a classic and beginner-friendly dataset that serves as an excellent starting point for understanding fundamental machine learning concepts, especially classification models like Multi-Layer Perceptrons (MLPs). It contains a total of **150 samples** across **3 flower classes**, each described by **4 numerical features**:

- 🌿 Sepal Length (cm)  
- 🌿 Sepal Width  (cm)  
- 🌸 Petal Length (cm)  
- 🌸 Petal Width  (cm)

## Methodology  

### 🔧 Data Preprocessing  
- Label encoding of flower species  
- Feature scaling using `StandardScaler` from `scikit-learn`

### 🧠 Model Architecture (MLP)  
- Input Layer: 4 nodes  
- Hidden Layer 1: 10 neurons, ReLU activation  
- Hidden Layer 2: 8 neurons, ReLU activation  
- Output Layer: 3 neurons, Softmax activation

### 🏋 Training  
- Optimizer: `Adam`  
- Loss Function: `CategoricalCrossentropy`
