import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

# Stratified sampling
class StratifiedSampling:
    def __init__(self, data):
        self.data = data
        self.X = self.data.iloc[:, :30].values
        self.y = self.data.iloc[:, 30:32].values
        self.stratify_label = None
        self.train_data = None
        self.test_data = None

    def stratify_data(self):
        """Using K-means and quantile methods"""

        # Creating labels for categorical variables
        ys_labels = ['Q1', 'Q2', 'Q3']
        elongation_labels = ['Q1', 'Q2', 'Q3']

        # Creating Categorical Variables
        self.data['YS_Category'] = pd.qcut(self.data["YS (MPa)"], q=3, labels=ys_labels, duplicates='drop')
        self.data['Compressive strain_Category'] = pd.qcut(self.data['Compressive strain (%)'], q=3, labels=elongation_labels,
                                                   duplicates='drop')

        # K-means
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.data.iloc[:, :16])
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.data['Cluster_Labels'] = kmeans.fit_predict(X_scaled)

        # Creating Layered Labels
        self.data['Stratify_Label'] = self.data['YS_Category'].astype(str) + '_' + self.data[
            'Compressive strain_Category'].astype(str) + '_' + self.data['Cluster_Labels'].astype(str)

        self.train_data, self.test_data = train_test_split(self.data, test_size=0.2,
                                                           stratify=self.data['Stratify_Label'], random_state=42)

        return self.train_data, self.test_data

    def stratified_kfold(self, n_splits=5):

        """Perform stratified cross-validation"""

        X = self.data.iloc[:, :30].values
        y_labels = self.data['Stratify_Label'].values
        y_true = self.data.iloc[:, 30:32].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X, y_labels):
            X_train, X_test = X[train_index], X[test_index]
            y_train_true, y_test_true = y_true[train_index], y_true[test_index]

            yield X_train, y_train_true, X_test, y_test_true

