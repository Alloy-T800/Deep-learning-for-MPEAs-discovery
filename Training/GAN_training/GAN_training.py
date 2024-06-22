from src.GAN_model.GAN_achi import Generator, Discriminator, DataVisualizer
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

# Checking GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Configuration File
with open('config.json', 'r') as f:
    config = json.load(f)

if not os.path.exists('Save_fig'):
    os.makedirs('Save_fig')
if not os.path.exists('Save_model'):
    os.makedirs('Save_model')
if not os.path.exists('Save_DATE'):
    os.makedirs('Save_DATE')

# Define the save path
# loss_log_path = 'GAN_loss_log.xlsx'
# d_losses = []
# g_losses = []

# Read data set
data = pd.read_excel("data_set_ori.xlsx")
features = data.iloc[:, :16].values

# Normalization
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Get the parameters of the normalization
data_min = torch.tensor(scaler.data_min_, dtype=torch.float32)
data_max = torch.tensor(scaler.data_max_, dtype=torch.float32)

# Max-min normalization function
def minmax_scaling(data, data_min, data_max, scaled_min=0, scaled_max=1):
    # Perform min-max scaling
    scaled_data = (data - data_min) / (data_max - data_min) * (scaled_max - scaled_min) + scaled_min
    return scaled_data

# The data is stored in the features_scaled variable
data_tensor = torch.FloatTensor(features_scaled)
data_loader = DataLoader(TensorDataset(data_tensor), batch_size=64, shuffle=True)

# Initialization Model
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define the optimizer and loss function
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.00015, betas=(0.5, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training settings
num_epochs = 200000
latent_dim = 100  # Noise dimension

# Model training
for epoch in range(num_epochs):
    for i, (real_data,) in enumerate(data_loader):
        batch_size = real_data.size(0)

        # Creating labels
        real_labels = (torch.ones(batch_size, 1) * 0.9).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Training discriminator
        optimizer_d.zero_grad()

        # Use the real data
        real_data = real_data.to(device)
        real_output = discriminator(real_data)
        d_loss_real = criterion(real_output, real_labels)

        # Use the generated data
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data_D = generator(noise).detach()  # Detach
        fake_data_100 = fake_data_D * 100
        fake_data_scaled = minmax_scaling(fake_data_100, data_min, data_max)

        fake_output = discriminator(fake_data_scaled.detach())  # Detach
        d_loss_fake = criterion(fake_output, fake_labels)

        # Backpropagation and optimization
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Training generator
        optimizer_g.zero_grad()

        # Using the generated data, the gradient is preserved
        fake_data_G = generator(noise)
        fake_data_G_100 = fake_data_G * 100
        fake_data_G_100_scaled = minmax_scaling(fake_data_G_100, data_min, data_max)

        output = discriminator(fake_data_G_100_scaled)
        g_loss = criterion(output, real_labels)

        # Backpropagation and optimization
        g_loss.backward()
        for name, param in generator.named_parameters():
            if param.grad is None:
                print(f"No gradient for {name}!")
        optimizer_g.step()

        # Print Loss
        if (epoch + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], d_loss: {d_loss.item():.8f}, g_loss: {g_loss.item():.8f}')

        # Every 500 epochs, samples are generated and visualized using a generator
        if epoch % 5000 == 0:
            noise = torch.randn(1000, latent_dim).to(device)
            with torch.no_grad():
                generated_samples = generator(noise).cpu().numpy()
                generated_data_100 = generated_samples * 100

            # Calling Visualization Functions
            visualizer = DataVisualizer()
            visualizer.visualize_distributions(generated_data_100, features, epoch)

        # Every 1000 epochs, save the model
        if epoch % 5000 == 0:
            torch.save(generator.state_dict(), f'Save_model/GAN_generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'Save_model/GAN_discriminator_epoch_{epoch}.pth')
            # Use the generator to generate 300 samples and save them
            noise = torch.randn(300, latent_dim).to(device)
            with torch.no_grad():
                generated_samples = generator(noise).cpu().numpy()
                generated_data_100 = generated_samples * 100

            # Save the generated data
            pd.DataFrame(generated_data_100).to_excel(f'Save_DATE/generated_data_epoch_{epoch}.xlsx', index=False)

        # # Save Loss
        # d_losses.append(d_loss.item())
        # g_losses.append(g_loss.item())
        #
        # if epoch % 1000 == 0:
        #     if os.path.exists(loss_log_path):
        #         loss_df = pd.read_excel(loss_log_path)
        #         new_data = pd.DataFrame({
        #             'epoch': range(epoch - len(d_losses) + 1, epoch + 1),
        #             'd_loss': d_losses,
        #             'g_loss': g_losses
        #         })
        #         loss_df = pd.concat([loss_df, new_data], ignore_index=True)
        #     else:
        #         loss_df = pd.DataFrame({
        #             'epoch': range(epoch + 1),
        #             'd_loss': d_losses,
        #             'g_loss': g_losses
        #         })
        #     # Data preservation
        #     loss_df.to_excel(loss_log_path, index=False)
        #
        #     d_losses = []
        #     g_losses = []

# Generate data
num_samples = 1000
noise = torch.randn(num_samples, latent_dim).to(device)
generated_data = generator(noise).detach().cpu().numpy()
generated_data_100 = generated_data * 100

# Save data
pd.DataFrame(generated_data_100).to_excel('GAN_generated_data.xlsx', index=False)

# Saving Model
torch.save(generator.state_dict(), 'GAN_generator.pth')
torch.save(discriminator.state_dict(), 'GAN_discriminator.pth')

# Save normalization parameters
scaler_params = {
    'min': scaler.data_min_.tolist(),
    'max': scaler.data_max_.tolist(),
}

with open('GAN_scaler_params.json', 'w') as f:
    json.dump(scaler_params, f)

