import torch
import torch.nn as nn
import torch.optim as optim
from MLP import MLP # Import the MLP model
from data import generateData, preprocessData # Import data functions

# Step 1: Load Data
n = 3 # Matrix size (n x n)
num_samples = 100000 # Number of matrices

# Generate and preprocess data
x, y = generateData(num_samples, n)
x_train, x_test, y_train, y_test = preprocessData(x, y)

# Check shapes of 'x_train' and 'y_train'
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

#Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

#Step 2: Initialize Model, Loss, and Optimizer
input_size = n * n
hidden_size = 256
output_size = n * n

model = MLP(input_size, hidden_size, output_size)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0003) # Adam Optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

# Early stopping values
best_loss = float('inf')
patience = 20
trigger_times = 0

# Step 3: Training Loop
epochs = 300
batch_size = 64

# Convert dataset to PyTorch DataLoader for batch processing
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

print("Starting Training...")

for epoch in range(epochs):
    
    total_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        
        optimizer.zero_grad() # Reset gradients
        
        outputs = model(batch_x) # Forward pass
        loss = criterion(outputs, batch_y) # Compute loss
        
        loss.backward() # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Update weights
        
        total_loss += loss.item()
        
    scheduler.step() # Adjust learning rate
    
    # Calculate average loss per epoch
    avg_loss = total_loss / len(train_loader)
        
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()}")
        
    # Early Stopping Check
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0 # Reset patience counter when improvement occurs
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
print("Training Complete!")
