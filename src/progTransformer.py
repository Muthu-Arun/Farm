import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the CSV data
df = pd.read_csv('/home/arun/dev/dataSets/crop_production.csv')

# Step 1: Preprocess the categorical features
label_encoders = {}
categorical_columns = ['State_Name', 'District_Name',
                       'Crop', 'Crop_Year', 'Season', 'Area', 'Production']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 2: Preprocess numerical features (Area, Production, Yield)
numerical_columns = ['Area', 'Production']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 3: Prepare the data for training
X = df[categorical_columns + numerical_columns].values
y = df['Production'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 4: Define the Transformer Model for Tabular Data


class TransformerTabularModel(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, dim_feedforward=128):
        super(TransformerTabularModel, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, input_dim, d_model))  # Fixed shape

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output layer for regression

    def forward(self, x):
        # Shape (batch_size, seq_length, d_model)
        x = self.embedding(x).unsqueeze(1)
        # Adjust positional encoding shape
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)  # Apply Transformer Encoder
        x = x.mean(dim=1)  # Aggregate across sequence dimension
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the model
input_dim = X_train.shape[1]
model = TransformerTabularModel(input_dim)

# Step 5: Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Training the model
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
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
    for i in range(5):
        print(f'Predicted: {test_predictions[i].item():.4f}, Actual: {
              y_test[i]:.4f}')
