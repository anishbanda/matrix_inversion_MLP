import torch
import torch.nn as nn
import numpy as np
from MLP import MLP # Import trained model class
from data import generateData, preprocessData # Import data functions
from train import model, scaler_x, scaler_y # Import trained model and scalers

# Step 1: Generate and Preprocess Test Data
n = 3 # Matrix size (n x n)
num_test_samples = 2000 # Number of test samples

x, y = generateData(num_test_samples, n)
x_train, x_test, y_train, y_test, scaler_x, scaler_y = preprocessData(x, y)

# Convert test data to PyTorch tensors
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Step 2: Run Model on Test Data
model.eval() # Set model to evaluation mode

with torch.no_grad(): # Disable gradient computation
    
    y_pred_tensor = model(x_test_tensor) # Get predictions
    
# Convert predictions and true values back to numpy arrays
y_pred = y_pred_tensor.numpy()
x_test = x_test_tensor.numpy()

# Step 3: Compute Evaluation Metrics
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error on Test Data: {mse:.6f}")

# Step 4: Compare a Few Predictions with Actual Inverses
num_samples_to_display = 3
for i in range(num_samples_to_display):
    
    original_matrix = x_test[i].reshape(n, n) # Reshape back to original form
    actual_inverse = y_test[i].reshape(n, n) # Reshape back to original form
    predicted_inverse = y_pred[i].reshape(n, n) # Reshape back to original form
    
    print("\nOriginal Matrix:")
    print(original_matrix)
    print("\nActual Inverse:")
    print(actual_inverse)
    print("\nPredicted Inverse")
    print(predicted_inverse)
    