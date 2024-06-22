import torch
import torch.nn as nn
import json

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration files
with open('config_ffn.json', 'r') as f:
    config = json.load(f)

# Define feature extraction units
class Feature_extra_units(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_units, dropout_rate, output_features):
        super(Feature_extra_units, self).__init__()

        # The hidden layers
        layers = []
        # Add the first layer to transform input_dim to hidden_units
        layers.append(nn.Linear(input_dim, hidden_units))  # Initialize the first layer with input_dim
        layers.append(nn.BatchNorm1d(hidden_units))  # Batch normalization
        layers.append(nn.ReLU())  # Activation
        layers.append(nn.Dropout(dropout_rate))  # Dropout

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_units, hidden_units),  # Fully connected layer
                nn.BatchNorm1d(hidden_units),  # Batch normalization
                nn.ReLU(),  # Activation function
                nn.Dropout(dropout_rate)  # Dropout
            ])

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_units, output_features)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# Defining the integrated decision unit
class Integrated_unit(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_units, dropout_rate):
        super(Integrated_unit, self).__init__()

        # Hidden layers
        layers = []
        # Add the first layer to transform input_dim to hidden_units
        layers.append(nn.Linear(input_dim, hidden_units))  # Initialize the first layer with input_dim
        layers.append(nn.BatchNorm1d(hidden_units))  # Batch normalization
        layers.append(nn.ReLU())  # Activation
        layers.append(nn.Dropout(dropout_rate))  # Dropout

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_units, hidden_units),  # Fully connected layer
                nn.BatchNorm1d(hidden_units),  # Batch normalization
                nn.ReLU(),  # Activation function
                nn.Dropout(dropout_rate)  # Dropout
            ])

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_units, 2)  # Assuming two outputs

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# Define the AFN model
class AFN_Model(nn.Module):
    def __init__(self, feature_extra_units, integrated_unit, num_feature_extra_units):
        super(AFN_Model, self).__init__()
        self.feature_extra_units = nn.ModuleList(feature_extra_units)
        self.integrated_unit = integrated_unit
        self.num_feature_extra_units = num_feature_extra_units

    def forward(self, x):
        feature_extra_units_outputs = [model(x) for model in self.feature_extra_units]
        integrated_unit_input = torch.cat(feature_extra_units_outputs, dim=1)
        return self.integrated_unit(integrated_unit_input)

# Model initialization
class FFNModelInitializer:
    def __init__(self, num_feature_extra_units, device, config):
        self.num_feature_extra_units = num_feature_extra_units
        self.device = device
        self.config = config
        self.input_features_Integrated_unit = num_feature_extra_units * config["output_features_feature_extra"]
        self.input_feature_extra = config["input_size"]

    def create_afn_model(self):

        # Hyperparameters of the feature extraction units
        feature_extra_units_params = {
            "input_dim": self.input_feature_extra,
            "num_hidden_layers": self.config["num_hidden_layers_feature_extra"],
            "hidden_units": self.config["hidden_units_feature_extra"],
            "dropout_rate": self.config["dropout_rate_feature_extra"],
            "output_features": self.config["output_features_feature_extra"],
        }

        feature_extra_models = [Feature_extra_units(**feature_extra_units_params) for _ in range(self.num_feature_extra_units)]

        # Hyperparameters of the integrated decision unit
        Integrated_unit_params = {
            "input_dim": self.input_features_Integrated_unit,
            "num_hidden_layers": self.config["num_hidden_layers_integrated_unit"],
            "hidden_units": self.config["hidden_units_integrated_unit"],
            "dropout_rate": self.config["dropout_rate_integrated_unit"]
        }
        Integrated_unit_model = Integrated_unit(**Integrated_unit_params)

        afn_model = AFN_Model(feature_extra_models, Integrated_unit_model, self.num_feature_extra_units)
        return afn_model.to(self.device)

# Calling the trained model
class ModelLoader:
    def __init__(self):
        self.AFN_model = None

    def load_model(self, model_path):
        # Loading Models
        self.AFN_model = torch.load(model_path, map_location=torch.device('cpu'))

        # Ensure that the model is in evaluation mode
        self.AFN_model.eval()

        if hasattr(self.AFN_model, "attention"):
            self.AFN_model.attention.eval()

        self.AFN_model.integrated_unit.eval()

        for units in self.AFN_model.feature_extra_units:
            units.eval()

    def get_model(self):
        return self.AFN_model
