import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """_summary_

    Args:
        input_size: Batch_size, channel_size, x, y , z... eg.. [64,1,16,16,16]
    """
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
        nn.Conv3d(input_size[1], 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm3d(64),
        nn.ReLU(),
        nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm3d(128),
        nn.ReLU(),
        nn.Conv3d(128, 256, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm3d(256),
        nn.ReLU(),
        )
        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x