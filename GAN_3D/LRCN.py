import torch
import torch.nn as nn
import torch.nn.functional as F

class LRCNModel(nn.Module):
    def __init__(self, dl, dh, lstm_hidden_size=200, lstm_layers=1, num_classes=1):
        super(LRCNModel, self).__init__()
        c = 5
        
        # 3D CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=c, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Flatten(),
            nn.Linear(256 * (dl // 4) * (dl // 4) * (c // 4), 200)  # Adjust the size according to your input/output
        )
        
        # LSTM
        self.lstm = nn.LSTM(input_size=200, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        
        # 2D Fully-Convolutional Decoder Network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=lstm_hidden_size, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, num_classes, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()
        # Process each volume through the encoder
        encoded_volumes = [self.encoder(x[:, i]) for i in range(seq_len)]
        encoded_sequence = torch.stack(encoded_volumes, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(encoded_sequence)
        
        # Decode each timestep output from LSTM
        decoded_images = [self.decoder(lstm_out[:, i].view(batch_size, -1, 1, 1)) for i in range(seq_len)]
        output_sequence = torch.stack(decoded_images, dim=1)
        
        return output_sequence
