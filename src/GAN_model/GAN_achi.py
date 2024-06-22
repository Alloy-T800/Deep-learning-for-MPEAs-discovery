import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Define the adjustment function
def adjusted_function(x, threshold=0.02):

    x_adjusted = x.clone()

    # Set elements less than the threshold value to 0
    x_adjusted[x_adjusted < threshold] = 0

    # Avoid dividing by 0
    sum_values = x_adjusted.sum(dim=1, keepdim=True)
    sum_values[sum_values == 0] = 1

    # Renormalize values
    x_adjusted /= sum_values

    return x_adjusted

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 224),
            nn.BatchNorm1d(224),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(224, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 36),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(36, 16),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        x_adjusted = adjusted_function(x)
        return x_adjusted

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Save Images
class DataVisualizer:
    def __init__(self, feature_names=None):
        if feature_names is None:
            self.feature_names = ['Fe', 'Co', 'Ni', 'Cr', 'Mn', 'Al', 'Cu', 'Ti', 'Zr', 'Nb', 'V', 'Mo', 'Hf', 'Ta', 'Si', 'W']
        else:
            self.feature_names = feature_names

    def visualize_distributions(self, generated_samples, real_data, epoch):

        if isinstance(generated_samples, torch.Tensor):
            generated_samples = generated_samples.detach().cpu().numpy()
        if isinstance(real_data, torch.Tensor):
            real_data = real_data.detach().cpu().numpy()

        # Compare the probability that each feature is 0
        plt.bar(range(generated_samples.shape[1]),
                np.mean(generated_samples == 0, axis=0),
                alpha=0.5,
                label='Generated',
                color='red')
        plt.bar(range(real_data.shape[1]),
                np.mean(real_data == 0, axis=0),
                alpha=0.5,
                label='Real',
                color='blue')
        plt.legend()
        plt.title(f'Probability of Zero per Feature at epoch {epoch}')
        plt.xticks(range(len(self.feature_names)), labels=self.feature_names)
        plt.tight_layout()
        plt.savefig(f'Save_fig/feature_0_{epoch}.png')
        plt.close()

        # characteristic nonzero distribution
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        for i in range(16):
            ax = axes[i]
            combined_data = np.concatenate([real_data[:, i], generated_samples[:, i]])
            bins = np.histogram_bin_edges(combined_data, bins='auto')
            ax.hist(real_data[:, i], bins=bins, alpha=0.5, label='Real', color='blue', density=True)
            ax.hist(generated_samples[:, i], bins=bins, alpha=0.5, label='Generated', color='red', density=True)
            ax.set_title(self.feature_names[i])
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'Save_fig/characteristic_nonzero_{epoch}.png')
        plt.close()

        # Non-zero feature number
        plt.figure(figsize=(10, 5))
        plt.hist(np.sum(generated_samples > 0, axis=1),
                 alpha=0.5,
                 bins=16,
                 label='Generated',
                 color='red',
                 density=True)
        plt.hist(np.sum(real_data > 0, axis=1),
                 alpha=0.5,
                 bins=16,
                 label='Real',
                 color='blue',
                 density=True)
        plt.legend()
        plt.title(f'Distribution of Non-zero Feature Count at epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'Save_fig/Non-zero_feature_number_{epoch}.png')
        plt.close()

# Calling the trained GAN
class GANLoader:
    def __init__(self, generator_path, discriminator_path):

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.generator.load_state_dict(torch.load(generator_path))
        self.discriminator.load_state_dict(torch.load(discriminator_path))

        self.generator.eval()
        self.discriminator.eval()

    def generate_data(self, num_samples, latent_dim=100):
        """
        Use the generator to generate the specified amount of data

        :parameter:
            num_samples (int): Number of samples to be generated
            latent_dim (int): Hidden space dimension, default 100

        :return:
            Tensor: Generated data
        """
        # Generate randomized potential vectors
        noise = torch.randn(num_samples, latent_dim)
        # Using generator to produce data
        return self.generator(noise)

    def discriminate(self, data):
        """
        Evaluating data using discriminator

        parameter:
            data (Tensor): Data to be assessed

        return:
            Tensor: Output of the discriminator
        """
        return self.discriminator(data)



