from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading and processing data
data = pd.read_excel("data_set_ori.xlsx")

# Max-min normalization
scaler = MinMaxScaler()
features = data.iloc[:, :30]
targets = data.iloc[:, 30:32]

scaled_features = scaler.fit_transform(features)
scaled_targets = scaler.fit_transform(targets)

# Converting data to NumPy arrays
X = np.array(scaled_features)
Y = np.array(scaled_targets)

# Leave One Out
loo = LeaveOneOut()
y_true, y_pred = [], []

# Modeling
model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=375, max_depth=2, learning_rate=0.04, random_state=None))

# The training process
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Training model
    model.fit(X_train, y_train)

    # Assessment model
    y_test_pred = model.predict(X_test)

    y_true.append(y_test)
    y_pred.append(y_test_pred)

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