import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Discriminator(nn.Module):
    """_summary_

    Args:
        input_size: Batch_size, channel_size, x, y , z... eg.. [64,1,16,16,16]
    """
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        # Dynamically compute the flat features after convolutions
        self._to_linear = self._get_conv_output(input_size)

        self.fc = nn.Linear(self._to_linear, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_conv_output(self, input_size):
        with torch.no_grad():  # Ensure no gradients are computed during this process
            input = torch.rand(*input_size)
            output = self.conv1(input)
            output = self.bn1(output)  # Apply batch normalization
            output = F.relu(output)  # Apply ReLU
            output = self.conv2(output)
            output = self.bn2(output)  # Apply batch normalization
            output = F.relu(output)  # Apply ReLU
            output = self.conv3(output)
            output = self.bn3(output)  # Apply batch normalization
            output = F.relu(output)  # Apply ReLU
            return int(np.prod(output.size()[1:]))  # Exclude batch size from the product calculation

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
