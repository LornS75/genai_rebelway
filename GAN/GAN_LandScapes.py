import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,utils
from torchvision.utils import save_image 


image_size = 64
channels_img = 3
z_dim = 100
features_g = 64
features_d = 64
batch_size = 128
lr = 2e-4
num_epochs = 50
device = torch.device("cuda")

transform = transforms.Compose([
    transforms.Resize(),
    transforms.CenterCrop(),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)])
])

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 8, 4, 1, 0),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g * 1, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.net(x)
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self,x):
        return self.net(x)
