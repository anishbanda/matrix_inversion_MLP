# Matrix Inversion using an MLP

This project implements a Multi-Layer Perceptron (MLP) to approximate
the matrix inversion function. The goal is to train a neural network to take an invertible matrix as input and predict its inverse.

##Learning Objectives

- Neural network function approximation
- Handling structures data (matrices)
- Optimization and loss functions in deep learning

---

## Installation

### Clone the Repository

```sh
git clone https://github.com/anishbanda/matrix_inversion_MLP.git
cd matrix_inversion_MLP

```

---

## Understanding the MLP

### 1. Problem Definition

The goal is to train an MLP to learn the function:
$f(A) = A^{-1}$
where A is an 'n x n' invertible matrix.

### 2. Neural Network Architecture

- **Input Layer**: Flattened matrix of size 'n^2'
- **Hidden Layers**: Fully connected layers with ReLU activations
- **Output Layer**: Flattened inverse matrix of size 'n^2'
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

## Training the Model

Run the training script with:

```sh
python train.py
```

This will:

- Generate random invertible matrices
- Train the MLP using PyTorch
- Save the trained model for future use

---

## Testing the Model

Once trained, you can test the model with:

```sh
python test.py
```

This script:

- Loads a trained MLP
- Generates a random matrix
- Predicts its inverse
- Compares it with the actual inverse

---

## Future Improvements

- Improve accuracy for larger matrices
- Experiment with different activation functions
- Test different loss functions
- Try CNNs or Transformer-based architectures for structured data
