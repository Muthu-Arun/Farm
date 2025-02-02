import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the CSV data
df = pd.read_csv('/home/arun/dev/dataSets/crop_production.csv')

# Step 1: Preprocess the categorical features
label_encoders = {}
categorical_columns = ['State_Name', 'District_Name',
                       'Crop', 'Crop_Year', 'Season', 'Area', 'Production']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders to decode if needed later

# Step 2: Preprocess numerical features (Area, Production, Yield)
numerical_columns = ['Area', 'Production']

# Normalize numerical features
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 3: Prepare the data for training
# All columns except the target
X = df[categorical_columns + numerical_columns].values
y = df['Production'].values  # Assuming 'Production' is the target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension (1 channel)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension (1 channel)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 4: Define the CNN Model for Tabular Data
class CNNTabularModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNTabularModel, self).__init__()

        # 1D Convolutional layer (kernel size of 3, stride 1, padding 1)
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool1d(2)

        # Calculate output dimension after convolution and pooling
        # Assuming input_dim is the number of features (columns)
        conv_output_dim = input_dim  # Start with input size
        for _ in range(2):  # Apply 2 conv layers
            conv_output_dim = (conv_output_dim + 2 * 1 - 3) // 1 + 1  # Convolution
            conv_output_dim = conv_output_dim // 2  # After pooling (divided by 2)
        
        # Fully connected layers after convolution
        self.fc1 = nn.Linear(32 * conv_output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output layer for predicting Production (regression)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply conv1 and pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Apply conv2 and pooling
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = torch.relu(self.fc1(x))  # Apply fully connected layers
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer
        return x

# Initialize the model
input_dim = X_train.shape[1]  # Number of features in the input
model = CNNTabularModel(input_dim)

# Step 5: Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training the model
epochs = 100
for epoch in range(epochs):
    model.train()

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    # Forward pass
    predictions = model(X_train_tensor)

    # Compute the loss
    loss = criterion(predictions, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 7: Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

    # Example: Print some predictions alongside the actual values
    for i in range(5):
        print(f'Predicted: {test_predictions[i].item():.4f}, Actual: {y_test[i]:.4f}')
