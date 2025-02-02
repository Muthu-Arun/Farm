import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample Data (Assuming 1D sensor readings over time)
num_features = 3  # Example: pH, moisture, temperature
seq_length = 10   # Time steps per sample
num_samples = 100000  # Dataset size

# Simulated dataset
X = np.random.rand(num_samples, num_features, seq_length).astype(
    np.float32)  # (batch, channels, seq_length)
# Regression target (soil quality score)
y = np.random.rand(num_samples, 1).astype(np.float32)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# CNN Model for Time-Series Data


class CNN1D(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * (seq_length // 2), 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Model Initialization
model = CNN1D(input_channels=num_features, output_size=1)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
