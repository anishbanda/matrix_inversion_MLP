import numpy as np
from sklearn.model_selection import train_test_split

def generateData(num_samples, n, condition_number_range=(1, 10)):
    
    matrices = []
    inverses = []
    
    while len(matrices) < num_samples:
        A = np.random.randn(n, n)
        u, s, vh = np.linalg.svd(A)
        
        # Control singular values to set condition number
        min_singular = 1
        max_singular = condition_number_range[1]
        s = np.linspace(min_singular, max_singular, n)
        A = np.dot(u * s, vh)
        
        try:
            A_inv = np.linalg.inv(A)
            matrices.append(A.flatten())
            inverses.append(A_inv.flatten())
        except np.linalg.LinAlgError:
            continue # Skip singular matrices
        
        return np.array(matrices, dtype=np.float32), np.array(inverses, dtype=np.float32)
    
def preprocessData(x, y):
    if (len(x) < 5):
        raise ValueError("Not enough samples for splitting. Check data generation")
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