# Understanding the Problem

I am training a Multi-Layer Perceptron (MLP) to approximate the matrix inversion function:

$f(A) = A^(-1)$

Where A is an ‘n $x$ n’ invertible matrix
The model takes A as input and predicts $A^(-1)$ as output

$f(A) = A^{-1}$

Where A is an ‘n x n’ invertible matrix
The model takes A as input and predicts $A^{-1}$ as output

#### To train this model, I need:

- A dataset (random invertible matrices + their inverses)
- A neural network model to learn the function
- A loss function to measure how well the model is learning
- An optimizer to adjust the model’s parameters during training

# Generating the Dataset

#### Before building the MLP, first generate the dataset consisting of:

- Random invertible matrices as input
- Their inverses as output

A function that generates random invertible matrices and their corresponding inverses:
**Steps**

- Generate a random ‘n x n’ matrix using NumPy
- Check if it’s invertible (i.e., det(A) != 0)
- Compute its inverse using np.linalg.inv(A)
- Store both the matrix and its inverse.

```sh
def generateData(numSamples, n):
    matrixList = []
    for i in range(numSamples):
        randMatrix = np.random.randn(n, n)
        if (np.linalg.det(randMatrix) != 0):
            invMatrix = np.linalg.inv(randMatrix)
        randMatrix = randMatrix.flatten()
        invMatrix = invMatrix.flatten()
        matrix =[randMatrix, invMatrix]
        matrixList.append(matrix)
```

# What Does Preprocessing Involve?

#### Splitting the Data

Divide the dataset into:

- Training Set (approx. 80%)
- Test Set (approx. 20%)

#### Normalization the Data

Since the values for the matrix vary widely, need to normalize them for better performance
A common technique used is standardization (zero mean, unit variance):
$X' = {X - μ}/σ$
Where μ is the mean and σ is the standard deviation of the dataset

This ensures that the values are well-distributed and help the network converge faster

# Building the MLP

**Steps**

- Define the MLP Architecture
- Choose the Activataion Functions
- Select the Loss Function & Optimizer
- Implement the Model in PyTorch
- Ensure the Model is Ready for Training

## Define the MLP Architecture

Since the input and output are flattened matrices of size $n^2$, the MLP needs to:

- Take an $n^2$-dimensional input
- Pass through multiple fully connected layers
- Output an $n^2$-dimensional prediction

#### Architecture Design

| Layer    | Type                   | Size  |
| -------- | ---------------------- | ----- |
| Input    | Fully Connected        | $n^2$ |
| Hidden 1 | Fully Connected        | 128   |
| Hidden 2 | Fully Connected + ReLU | 128   |
| Hidden 3 | Fully Connected + ReLU | 128   |
| Output   | Fully Connected        | $n^2$ |

## Choose Activation Functions

Using ReLU (Rectified Linear Unit) for hidden layers because:

- Helps prevent vanishing gradient problem
- Allows model to learn complex non-linear mappings

The output layer **does not use an activation function** because this is a **regression problem**, and we want raw continuous values.

## Select the Loss Function & Optimizer

Loss Function:

- Use Mean Squared Error (MSE) becuase we are performing regression
- MSE penalizes large errors more than small ones

Optimizer:

- Use Adam Optimizer, which adapts the learning rate during training for faster convergence

## Implement the Model in PyTorch

**Steps**

- Define an `MLP` class using `torch.nn.Module`
- Initialize the layers in the constructor `(__init__)`
- Define the forward pass in `forward(self, x)`
- Ensure the model works on sample data before training
