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
categorical_columns = ['State_Name', 'District_Name', 'Crop',
                       'Crop_Year', 'Season', 'Area', 'Production']
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
y = df['Production'].values  # Assuming 'Yield' is the target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 4: Define the model


class CropPredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(CropPredictionModel, self).__init__()

        # Embedding layers for categorical features
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer for predicting Yield (regression)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the model
input_dim = X_train.shape[1]  # Number of features in the input
model = CropPredictionModel(input_dim)

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
        print(f'Predicted: {test_predictions[i].item():.4f}, Actual: {
              y_test[i]:.4f}')
