import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Checking GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ANN class definition
class ANN(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_units, dropout_rate):
        super(ANN, self).__init__()

        # Dynamically create hidden layers
        layers = []
        # Add the first layer to transform input_dim to hidden_units
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.BatchNorm1d(hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))  # Fully connected layer
            layers.append(nn.BatchNorm1d(hidden_units))  # Batch normalization
            layers.append(nn.ReLU())  # Activation function
            layers.append(nn.Dropout(dropout_rate))  # Dropout

        # Sequential container for hidden layers
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_units, 2)  # Assuming 2 outputs for example

    def forward(self, x):
        # Forward pass through all hidden layers
        x = self.hidden_layers(x)

        # Forward pass through the output layer
        x = self.output_layer(x)
        return x


# Reading and processing data
data = pd.read_excel("data_set_ori.xlsx")
scaler = MinMaxScaler()
features = data.iloc[:, :30]
targets = data.iloc[:, 30:32]
scaled_features = scaler.fit_transform(features)
scaled_targets = scaler.fit_transform(targets)
X = torch.tensor(scaled_features, dtype=torch.float32).to(device)
Y = torch.tensor(scaled_targets, dtype=torch.float32).to(device)

# Leave-one-out training and testing
loo = LeaveOneOut()
y_true, y_pred = [], []

# Through a cyclic training and testing process
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Initializing the model and optimizer
    model = ANN(input_dim=30, num_hidden_layers=2, hidden_units=400, dropout_rate=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()

    # Training on multiple epochs
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_train_pred = model(X_train)
        loss = loss_fn(y_train_pred, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation Model
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        y_test_pred_unscaled = scaler.inverse_transform(y_test_pred.cpu().numpy())  # Inverse normalized predicted value
        y_test_unscaled = scaler.inverse_transform(y_test.cpu().numpy())  # Inverse normalized true value
        print(f"True Values = {y_test_unscaled.flatten()} Pred Values = {y_test_pred_unscaled.flatten()}")

    y_true.append(y_test.cpu().numpy())
    y_pred.append(y_test_pred.cpu().numpy())

    print(f"Processed test sample {len(y_true)}/{len(X)}")

# Inverse normalization and computational indicators
y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)
y_true_unscaled = scaler.inverse_transform(y_true)
y_pred_unscaled = scaler.inverse_transform(y_pred)

for i in range(y_true.shape[1]):
    r2 = r2_score(y_true_unscaled[:, i], y_pred_unscaled[:, i])
    rmse = np.sqrt(mean_squared_error(y_true_unscaled[:, i], y_pred_unscaled[:, i]))
    print(f"R2 Score for target {i+1}: {r2:.4f}")
    print(f"RMSE Score for target {i + 1}: {rmse:.4f}")

# Plotting Scatter Plots
plt.figure(figsize=(14, 6))

# Target 1
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.scatter(y_true_unscaled[:, 0], y_pred_unscaled[:, 0], color='blue', alpha=0.5)
plt.title('Scatter plot for Target 1')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.axis('equal')
plt.plot([y_true_unscaled[:, 0].min(), y_true_unscaled[:, 0].max()], [y_true_unscaled[:, 0].min(), y_true_unscaled[:, 0].max()], 'k--')

# Target 2
plt.subplot(1, 2, 2)
plt.scatter(y_true_unscaled[:, 1], y_pred_unscaled[:, 1], color='red', alpha=0.5)
plt.title('Scatter plot for Target 2')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.axis('equal')
plt.plot([y_true_unscaled[:, 1].min(), y_true_unscaled[:, 1].max()], [y_true_unscaled[:, 1].min(), y_true_unscaled[:, 1].max()], 'k--')
plt.show()

# Save predictions to Excel
result_df = pd.DataFrame(np.hstack((features, y_true_unscaled, y_pred_unscaled)), columns=list(features.columns) + ['True_Target1', 'True_Target2', 'Pred_Target1', 'Pred_Target2'])
result_df.to_excel("ANN_predictions_results.xlsx", index=False)
print("The results have been saved")

