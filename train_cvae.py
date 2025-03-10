# squeue -u $(whoami) # sacct -u $(whoami) # sbatch my_job_script.sh
import csv
import pandas as pd
import numpy as np
import pyreadr
import pyreadstat
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
import matplotlib.pyplot as plt
# from longitude_transform import get_longitudinal_map_each, sphere_to_grid_each, color_map_DK, plot_original, plot_DK_map
# from helper_func_prep import SOI_array_per, min_max_normalize, plot_SOI

# 
# read this saved file to tensor
# Load data from .npy file
np.set_printoptions(precision=20) 
loaded_tensor = np.load('sample_input_tensor.npy')
loaded_phenotype_tensor = np.load('sample_phenotype_tensor.npy')
# Convert numpy array to PyTorch tensor
input_tensor_ = torch.tensor(loaded_tensor, dtype=torch.float32)
subset_phenotype_tensor_ = torch.tensor(loaded_phenotype_tensor, dtype=torch.float32)
torch.set_printoptions(precision=20)
# print(input_tensor_, subset_phenotype_tensor_)
#
batch_size = 100
dataset_sub = TensorDataset(input_tensor_, subset_phenotype_tensor_)
dataloader_sub = DataLoader(dataset_sub, batch_size=batch_size, shuffle=True)


# new model for left half brain!
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim):
        super(ConditionalVAE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim + cond_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, latent_dim)
        self.fc32 = nn.Linear(256, latent_dim)
        self.fc4 = nn.Linear(latent_dim + cond_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, input_dim)
    # Initialize encoder weights using Xavier initialization
        # for layer in self.encoder:
        #     if isinstance(layer, nn.Linear):
        #         init.xavier_uniform_(layer.weight)
    def encode(self, x, c): # Q(z|x, c)
        x = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(x)) #hidden layer 1
        h2 = F.relu(self.fc2(h1))#hidden layer 2
        z_mu = self.fc31(h2)
        z_var = self.fc32(h2)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        z = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc4(z))#hidden layer 3
        h4 = F.relu(self.fc5(h3))#hidden layer 4
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 769*195*4), c)
        # mu, logvar = self.encode(x.view(x.size(0), -1), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = F.mse_loss(recon_x, x.view(-1, 769*195*4), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss, KLD

# Hyperparameters
num_phenotype_features = 5  # sum_att  sum_agg	age	sex	edu_maternal
input_dim = 769*195*4  # Adjusted for the input shape
latent_dim =128

# Initialize CVAE model
cvae = ConditionalVAE(input_dim, num_phenotype_features, latent_dim)


# Training loop: Initialize empty lists to store loss values for each type of loss
reconstruction_loss_values = []
kl_loss_values = []
total_loss_values = []
optimizer = optim.Adam(cvae.parameters(), lr=1e-5)#, weight_decay=1e-5
# Training loop
num_epochs = 21
with open('training_loss_cvae.csv', 'w', newline='') as csvfile:
    fieldnames = ['Epoch', 'Reconstruction Loss', 'KL Loss', 'Total Loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for epoch in range(num_epochs):
    reconstruction_epoch_loss = 0.0
    kl_epoch_loss = 0.0
    total_epoch_loss = 0.0
    
    for batch_input, batch_phenotype in dataloader_sub:
        optimizer.zero_grad()
        recon_batch, z_mean, z_log_var = cvae(batch_input.float(), batch_phenotype.float())
        reconstruction_loss, kl_loss = loss_function(recon_batch, batch_input.float(), z_mean, z_log_var)
        loss = reconstruction_loss + kl_loss
        loss.backward()
        optimizer.step()
        
        reconstruction_epoch_loss += reconstruction_loss.item()
        kl_epoch_loss += kl_loss.item()
        total_epoch_loss += loss.item()
    
    reconstruction_epoch_loss /= len(dataloader_sub)
    kl_epoch_loss /= len(dataloader_sub)
    total_epoch_loss /= len(dataloader_sub)
    
    reconstruction_loss_values.append(reconstruction_epoch_loss)
    kl_loss_values.append(kl_epoch_loss)
    total_loss_values.append(total_epoch_loss)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {reconstruction_epoch_loss:.4f}, KL Loss: {kl_epoch_loss:.4f}, Total Loss: {total_epoch_loss:.4f}")

    with open('training_loss_cvae.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Epoch': epoch+1, 'Reconstruction Loss': reconstruction_epoch_loss, 'KL Loss': kl_epoch_loss, 'Total Loss': total_epoch_loss})
    # print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {reconstruction_epoch_loss:.4f}, KL Loss: {kl_epoch_loss:.4f}, Total Loss: {total_epoch_loss:.4f}")

# Plot the loss values
# plt.plot(range(1, num_epochs+1), reconstruction_loss_values, label='Reconstruction Loss')
# plt.plot(range(1, num_epochs+1), kl_loss_values, label='KL Loss')
# plt.plot(range(1, num_epochs+1), total_loss_values, label='Total Loss')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Losses")
# plt.legend()
# plt.grid(True)
# plt.show()


