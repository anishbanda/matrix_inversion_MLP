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
def generateData(num_samples, n):

    matrices = []
    inverses = []

    while len(matrices) < num_samples:
        A = np.random.randn(n, n) # Generate random matrix
        try:
            A_inv = np.linalg.inv(A) # Attempt to invert
            matrices.append(A.flatten()) # Store as flattened array
            inverses.append(A_inv.flatten()) # Store inverse as flattened array
        except np.linalg.LinAlgError:
            pass

    return np.array(matrices, dtype=np.float32), np.array(inverses, dtype=np.float32)
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

```sh
def preprocessData(x, y):
    # Split into training and test sets
    x_train, x_test, y_test, y_train = train_test_split(x, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test) # Use same scaler fitted on training data

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    # Return the processed data and scalers
    return x_train, x_test, y_train, y_test, scaler_x, scaler_y
```

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

```sh
class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(MLP, self).__init__()

        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size) # First hidden layer
        self.fc2 = nn.Linear(input_size, hidden_size) # Second hidden layer
        self.fc3 = nn.Linear(input_size, hidden_size) # Third hidden layer
        self.fc4 = nn.Linear(input_size, output_size) # Output layer

        # Define activation function
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x)) # Apply ReLU to first hidden layer
        x = self.relu(self.fc2(x)) # Apply ReLU to second hidden layer
        x = self.relu(self.fc3(x)) # Apply ReLU to third hidden layer
        x = self.fc4(x) # Output layer (no activation)

        return x
```

# Training the MLP

**Steps**

- Load preprocessed data from `main.py`
- Define loss function (Mean Squared Error)
- Choose optimizer (Adam optimizer)
- Set up training loop:
  1. Forward pass: Predict matrix inverses
  2. Compute the loss
  3. Backpropagate the gradients
  4. Update model parameters
- Monitor training performances using loss values

## Loading the Preprocessed Data

Import data from `main.py` using the `generateData()` and `preprocessData()` functions

## Define the Training Setup

Loss Function:

```sh
criterion = nn.MSELoss()
```

Since this is a regression task, use MSE loss.

Optimizer

```sh
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Adam optimizer is common for neural networks
Learning Rate (`lr`) controls how much the model updates its weights.

## Implement the Training Loop

Training epoch performs the following:

- Forward pass: Feed input `x_train` into the MLP
- Compute the loss between predicted and actual inverses
- Backward pass: Compute gradients
- Update weights using optimizer
- Print loss every few epochs to monitor progress

# Evaluate the MLP

**Steps**

- Load trained model from `train.py`
- Use test dataset (x_test, y_test) to predict matrix inverses
- Calculate the Mean Squared Error (MSE) and other metrics
- Compare the predicted v.s. actual inverse matrices

## Load the Trained Model

Since the model is already trained in `train.py`,

- Load `x_test` and `y_test`
- Run predictions on `x_test`
- Compare with `y_test`

## Implement Evaluation Script

**Steps**

1. Generates fresh test data using `generateData` and `preprocessData`
2. Runs the trained model on `x_test` to predict inverses
3. Computes the MSE to evaluate performance
4. Displays matrices
   - Original matrix (input)
   - Actual inverse (y)
   - Predicted inverse (ouput)
