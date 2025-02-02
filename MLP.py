import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        super(MLP, self).__init__()
        
        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size) # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, hidden_size) # Third hidden layer
        self.fc4 = nn.Linear(hidden_size, output_size) # Output layer
        
        # Batch Normalization Layers
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        # self.bn4 = nn.BatchNorm1d(hidden_size)
        # self.bn5 = nn.BatchNorm1d(hidden_size)
        
        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.2) 
        
    def forward(self, x):
        
        residual = x # Save the input for residual connection
        
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x) 
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        
        x += residual
        
        return x
    
# Testing the model with dummy input
n = 3 # Matrix size (n x n)
input_size = n * n
hidden_size = 64
output_size = n * n

model = MLP(input_size, hidden_size, output_size)
dummy_input = torch.randn(1, input_size) # Create a random input
output = model(dummy_input) # Run through the model

print("Model Output Shape:", output.shape) # Should be (1, output_size)