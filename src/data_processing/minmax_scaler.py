import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json

# Load Configuration File
with open('config.json', 'r') as f:
    config = json.load(f)

# Load data set
data = pd.read_excel("data_set_ori.xlsx")

X = data.iloc[:, 1:31].values
y = data.iloc[:, 31:33].values

# Normalize x and y
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

normalized_X = X_scaler.fit_transform(X)
normalized_y = y_scaler.fit_transform(y)

# Save normalization parameters
joblib.dump(X_scaler, config['X_scaler_path'])
joblib.dump(y_scaler, config['y_scaler_path'])


class MinMax:
    def __init__(self):
        self.X_scaler = joblib.load(config['X_scaler_path'])
        self.y_scaler = joblib.load(config['y_scaler_path'])
    def transform(self, X, y):
        return self.X_scaler.transform(X), self.y_scaler.transform(y)
    def transform_X(self, X):
        return self.X_scaler.transform(X)
    def transform_y(self, y):
        return self.y_scaler.transform(y)

    def inverse_transform(self, normalized_X, normalized_y):
        return self.X_scaler.inverse_transform(normalized_X), self.y_scaler.inverse_transform(normalized_y)
    def inverse_transform_nor_X(self, normalized_X):
        return self.X_scaler.inverse_transform(normalized_X)
    def inverse_transform_nor_y(self, normalized_y):
        return self.y_scaler.inverse_transform(normalized_y)
