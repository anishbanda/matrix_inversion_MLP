import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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

# Warmup Scheduler
warmup_epochs = 10
initial_lr = 1e-5
target_lr = 3e-4

model = MLP(input_size, hidden_size, output_size)
criterion = nn.SmoothL1Loss(reduction='mean')

# Custom warmup + cosine annealing scheduler
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-6)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

# Early stopping values
best_loss = float('inf')
patience = 20
trigger_times = 0

# Step 3: Training Loop
epochs = 300
initial_batch_size = 32
max_batch_size = 256

# Convert dataset to PyTorch DataLoader for batch processing
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
batch_size = initial_batch_size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Starting Training...")

for epoch in range(epochs):
    
    total_loss = 0.0
    
    if (epoch % 50 == 0) and (epoch != 0) and (batch_size < max_batch_size):
        batch_size = min(batch_size * 2, max_batch_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"Batch size increased to {batch_size} at epoch {epoch + 1}")
    
    for batch_x, batch_y in train_loader:
        
        optimizer.zero_grad() # Reset gradients
        
        outputs = model(batch_x) # Forward pass
        loss = criterion(outputs, batch_y) # Compute loss
        
        loss.backward() # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Update weights
        
        total_loss += loss.item()
        
    if epoch < warmup_epochs:
        lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:   
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
