import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        super(MLP, self).__init__()
        
        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size) # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, hidden_size) # Third hidden layer
        self.fc4 = nn.Linear(hidden_size, hidden_size) # Fourth hidden layer
        self.fc5 = nn.Linear(hidden_size, hidden_size) # Fifth hidden layer
        self.fc6 = nn.Linear(hidden_size, output_size) # Output layer
        
        # Define activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.relu(self.fc1(x)) # Apply ReLU to first hidden layer
        x = self.relu(self.fc2(x)) # Apply ReLU to second hidden layer
        x = self.relu(self.fc3(x)) # Apply ReLU to third hidden layer
        x = self.relu(self.fc4(x)) # Apply ReLU to fourth hidden layer
        x = self.relu(self.fc5(x)) # Apply ReLU to fifth hidden layer
        x = self.fc6(x) # Output layer (no activation
        
        return x
    
# Testing the model with dummy input
n = 3 # Matrix size (n x n)
input_size = n * n
hidden_size = 256
output_size = n * n

model = MLP(input_size, hidden_size, output_size)
dummy_input = torch.randn(1, input_size) # Create a random input
output = model(dummy_input) # Run through the model

print("Model Output Shape:", output.shape) # Should be (1, output_size)