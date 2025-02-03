import numpy as np
from sklearn.model_selection import train_test_split

def generateData(num_samples, n):
    
    matrices = []
    inverses = []
    
    while len(matrices) < num_samples:
        A = np.random.randn(n, n) # Generate random matrix
        try:
            if np.abs(np.linalg.det(A)) > 0.5:
                A_inv = np.linalg.inv(A) # Attempt to invert
                matrices.append(A.flatten()) # Store as flattened array
                inverses.append(A_inv.flatten()) # Store inverse as flattened array
            else: pass
        except np.linalg.LinAlgError:
            pass
        
    return np.array(matrices, dtype=np.float32), np.array(inverses, dtype=np.float32)

def preprocessData(x, y):
    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Return the processed data
    return x_train, x_test, y_train, y_test

# Generate Data
x, y = generateData(1000, 3)

# Preprocess Data
x_train, x_test, y_train, y_test = preprocessData(x, y)

# Print Sample Outputs
print("Matrix Train Data:", x_train[0])
print("Matrix Test Data:", x_test[0])
print("Inverse Train Data:", y_train[0])
print("Inverse Test Data:", y_test[0])