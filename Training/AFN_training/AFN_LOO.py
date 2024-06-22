from src.AFN_model.AFN_model_archi import AFNModelInitializer
import torch
import json
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Load Configuration File
with open('config.json', 'r') as f:
    config = json.load(f)

# Checking GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read the file
data = pd.read_excel("data_set_ori.xlsx")

# Min-max normalization
scaler = MinMaxScaler()
features = data.iloc[:, :30]
targets = data.iloc[:, 30:32]

scaled_features = scaler.fit_transform(features)
scaled_targets = scaler.fit_transform(targets)

scaled_data = pd.DataFrame(np.hstack((scaled_features, scaled_targets)), columns=data.columns[:32])

# Convert data to Tensor
data_tensor = torch.Tensor(scaled_data.values).to(device)

# Model initialization
initializer = AFNModelInitializer(config["num_feature_extra_units"], device, config, use_attention=True)

# The loss function
loss_fn = nn.L1Loss()

loo = LeaveOneOut()
y_true, y_pred = [], []
total_samples = data_tensor.size(0)

# The training process
for i, (train_index, test_index) in enumerate(loo.split(data_tensor)):

    AFN_model = initializer.create_afn_model().to(device)
    optimizer = optim.Adam(AFN_model.parameters(), lr=config["lr"])

    X_train, X_test = data_tensor[train_index, :30], data_tensor[test_index, :30]
    y_train, y_test = data_tensor[train_index, 30:], data_tensor[test_index, 30:]

    print(f"Processing LOO iteration {i+1}/{total_samples}")

    # Training
    for epoch in range(config["epochs"]):
        AFN_model.train()
        optimizer.zero_grad()
        y_train_pred = AFN_model(X_train)
        train_loss = loss_fn(y_train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        # Printing training loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{config['epochs']}, Train Loss: {train_loss.item():.4f}")

    # Evaluation model
    AFN_model.eval()
    with torch.no_grad():
        y_test_pred = AFN_model(X_test)
        y_test_pred_unscaled = scaler.inverse_transform(y_test_pred.cpu().numpy())
        y_test_unscaled = scaler.inverse_transform(y_test.cpu().numpy())

        y_true.append(y_test_unscaled)
        y_pred.append(y_test_pred_unscaled)

