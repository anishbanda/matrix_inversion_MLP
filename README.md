# Understanding the Problem

I am training a Multi-Layer Perceptron (MLP) to approximate the matrix inversion function:

$f(A) = A^{-1}$

Where A is an ‘n $x$ n’ invertible matrix
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
