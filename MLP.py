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
        self.fc4 = nn.Linear(hidden_size, hidden_size) # Fourth hidden layer
        self.fc5 = nn.Linear(hidden_size, hidden_size) # Fifth hidden layer
        self.fc6 = nn.Linear(hidden_size, output_size) # Output layer
        
        self.dropout = nn.Dropout(0.2) # Dropout to reduce overfitting
        
    def forward(self, x):
        
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x) 
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x))
        x = self.fc6(x)
        
        return x
    
# Testing the model with dummy input
n = 3 # Matrix size (n x n)
input_size = n * n
hidden_size = 128
output_size = n * n

model = MLP(input_size, hidden_size, output_size)
dummy_input = torch.randn(1, input_size) # Create a random input
output = model(dummy_input) # Run through the model

print("Model Output Shape:", output.shape) # Should be (1, output_size)