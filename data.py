import numpy as np
from sklearn.model_selection import train_test_split

def generateData(num_samples, n, condition_number_range=(1, 10), max_attempts=100000):
    
    matrices = []
    inverses = []
    attempts = 0
    
    while len(matrices) < num_samples and attempts < max_attempts:
        A = np.random.randn(n, n)
        
        try:
            A_inv = np.linalg.inv(A)
            cond_num = np.linalg.cond(A)
            
            # Condition number filter
            if condition_number_range[0] <= cond_num <= condition_number_range[1]:
                matrices.append(A.flatten())
                inverses.append(A_inv.flatten())
                
        except np.linalg.LinAlgError:
            pass
        
        attempts += 1
        
    if len(matrices) < num_samples:
        print(f"Warning: Only {len(matrices)} samples generated after {attempts} attempts")
        
        if len(matrices) < 5000:
            print("Reducing batch size and epochs due to limited data")
            
            global curriculum
            curriculum = [
                {**phase, "batch_size": min(phase['batch_size'], 32), "epochs": 50}
                for phase in curriculum
            ]
            
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