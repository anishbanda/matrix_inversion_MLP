import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from MLP import MLP # Import the MLP model
from data import generateData, preprocessData # Import data functions

# Step 1: Load Data
n = 3 # Matrix size (n x n)
num_samples = 100000 # Number of matrices

# Generate and preprocess data
# x, y = generateData(num_samples, n)
# x_train, x_test, y_train, y_test = preprocessData(x, y)

# Check shapes of 'x_train' and 'y_train'
# print(f"x_train shape: {x_train.shape}")
# print(f"y_train shape: {y_train.shape}")

#Convert to PyTorch tensors
# x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

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
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Curriculum difficulty levels
curriculum = [
    {"condition_range": (1, 30), "epochs": 100, "batch_size": 64}, # Easy
    {"condition_range": (20, 150), "epochs": 100, "batch_size": 128}, # Medium
    {"condition_range": (130, 300), "epochs": 100, "batch_size": 256} # Hard
]

# Early stopping values
best_loss = float('inf')
patience = 50
trigger_times = 0

# Step 3: Training Loop
epochs = 300

print("Starting Curriculum Learning...")

for phase_idx, phase in enumerate(curriculum):
    
    print(f"\nPhase {phase_idx + 1}: Condition Number Range {phase['condition_range']}")
    
    # Generate data for the current phase
    x, y = generateData(num_samples, n, condition_number_range=phase['condition_range'])
    x_train, x_test, y_train, y_test = preprocessData(x, y)
    
    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoader with the phase's batch size
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=phase['batch_size'], shuffle=True)

    # Training for the current phase
    for epoch in range(phase['epochs']):
        
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{phase['epochs']}], Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()}")
            
        # Early Stopping Check
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
    with torch.no_grad():
        predictions = model(x_test_tensor)
        mse = nn.MSELoss()(predictions, y_test_tensor)
        print(f"Phase {phase_idx + 1} Evaluation - MSE: {mse.item():.6f}")
        
print("Curriculum Learning Complete!")