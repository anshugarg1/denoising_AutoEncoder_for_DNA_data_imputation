import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        
        # decoder layers
        self.dec1 = nn.ConvTranspose1d(8, 8, kernel_size=2, stride=2)  
        self.dec2 = nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose1d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose1d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv1d(64, 8, kernel_size=3)
        
    def forward(self, x):
        # encoder
#         print(x.shape)
        x = F.relu(self.enc1(x))
#         print(x.shape)
        x = self.pool(x)
#         print(x.shape)
        x = F.relu(self.enc2(x))
#         print(x.shape)
        x = self.pool(x)
#         print(x.shape)
        x = F.relu(self.enc3(x))
#         print(x.shape)
        x = self.pool(x)
#         print(x.shape)
        x = F.relu(self.enc4(x))
#         print(x.shape)
        x = self.pool(x) # the latent space representation
#         print(x.shape)
        
        # decoder
        x = F.relu(self.dec1(x))
#         print(x.shape)
        x = F.relu(self.dec2(x))
#         print(x.shape)
        x = F.relu(self.dec3(x))
#         print(x.shape)
        x = F.relu(self.dec4(x))
#         print(x.shape)
        x = torch.sigmoid(self.out(x))
#         print(x.shape)
        return x