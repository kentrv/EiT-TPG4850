import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LRCNModel(nn.Module):
    def __init__(self, dl, dh, lstm_hidden_size=200, lstm_layers=1, num_classes=1):
        super(LRCNModel, self).__init__()
        self.dh = dh
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * dl // 4 * dl // 4 * dl // 4, lstm_hidden_size)
        )

        self.lstm = nn.LSTM(input_size=lstm_hidden_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        
        # Dynamically create decoder layers to upscale to dh x dh
        self.decoder = self._create_decoder(lstm_hidden_size, num_classes)

    def _create_decoder(self, in_channels, out_channels):
        layers = []
        current_size = 1  # Starting from 1x1
        target_size = self.dh
        
        while current_size < target_size:
            next_size = min(current_size * 2, target_size)
            layers.append(nn.ConvTranspose2d(in_channels if current_size == 1 else 128, 128 if next_size != target_size else out_channels, kernel_size=4, stride=2, padding=1))
            if next_size != target_size:  # No BatchNorm or ReLU after the last layer
                layers.append(nn.BatchNorm2d(128))
                layers.append(nn.ReLU())
            current_size = next_size
            in_channels = 128  # After the first layer, input channels to the next layer would be 128
        
        layers.append(nn.Tanh())  # Tanh activation at the end
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        #x = x.view(batch_size * seq_len, c, h, w)
        x = self.encoder(x)
        # Reshape to get the sequence back for the LSTM
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        # Reshape LSTM output for decoding; ensuring it's 4D: [N, C, H, W]
        lstm_out = lstm_out.view(batch_size * seq_len, -1)  # This will be [batch_size * seq_len, lstm_hidden_size]
        lstm_out = lstm_out.unsqueeze(-1).unsqueeze(-1)  # Now [batch_size * seq_len, lstm_hidden_size, 1, 1]
        decoded_images = self.decoder(lstm_out)
        decoded_images = self.decoder(lstm_out)
        decoded_images = decoded_images.view(batch_size, seq_len, self.dh, self.dh) 

        return decoded_images

