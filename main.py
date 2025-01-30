import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Generate Data
x, y = generateData(1000, 3)

# Preprocess Data
x_train, x_test, y_train, y_test, scaler_x, scaler_y = preprocessData(x, y)

# Print Sample Outputs
print("Matrix Train Data:", x_train[0])
print("Matrix Test Data:", x_test[0])
print("Inverse Train Data:", y_train[0])
print("Inverse Test Data:", y_test[0])