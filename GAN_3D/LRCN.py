import torch
import torch.nn as nn
import torch.nn.functional as F

class LRCNModel(nn.Module):
    def __init__(self, dl, dh, num_slices=1, lstm_hidden_size=200, lstm_layers=1, num_classes=1):
        """_summary_

        Args:
            dl (int): the input 2D image size (dl x dl)
            dh (int): the output 3D image size (dh x dh x dh)
            c (int, optional): length of sequence. Defaults to 1.
            lstm_hidden_size (int, optional): size of vector created by lstm. Defaults to 200.
            lstm_layers (int, optional): how many layers the lstm should have. Defaults to 1.
            num_classes (int, optional): _description_. Defaults to 1.
        """
        super(LRCNModel, self).__init__()
        self.dl = dl
        self.dh = dh
        self.num_slices = num_slices
        self.lstm_hidden_size = lstm_hidden_size

        # Calculate the size of the flattened feature vector after Conv3d layers
        conv3d_output_size = 128 * (dl // 4) * (dl // 4) * (dl // 4)
        
        # Encoder: CNN to LSTM
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, lstm_hidden_size)
        )

        self.lstm = nn.LSTM(input_size=lstm_hidden_size, hidden_size=lstm_hidden_size, num_layers=num_slices, batch_first=True, dropout=0.5)
            
        # Project LSTM outputs to a higher-dimensional space
        projected_dim = dh * dh  # Adjust this based on the decoder architecture
        self.decoder_fc = nn.Linear(lstm_hidden_size, projected_dim * num_slices)
    
        # Decoder: LSTM to CNN
        # Assuming `dh` is the desired spatial dimension of the output image. Adjust `conv2d_input_size` based on the decoder architecture.
                
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(5,5), stride=(2,2), padding=(2,2), output_padding=(1,1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        batch_size, num_slices, l, h, w = x.size()

        # num slices represents a mini-batch
        # each slice is a sequence of 2d images (h x w) of length l
        
        x_reshaped = x.view(batch_size * num_slices, 1, l, h, w)  # Add a channel dimension

        # Encoder
        encoded = self.encoder(x_reshaped)

        # for LSTM,
        lstm_input = encoded.view(batch_size, num_slices, -1)
        # LSTM out should be [batch, num_slices, features]
        lstm_out, _ = self.lstm(lstm_input)
        
        
        # Decoder FC and reshape for decoder
        decoded = self.decoder_fc(lstm_out)
        
        
        # Decoder should take [batch, num_slices, features] as input and produce [batch, num_slices, dh, dh]
        decoded = decoded.view(batch_size * num_slices, self.num_slices, self.dh, self.dh)  # Adjust as needed

        # Decoder to reconstruct slices: adjust if decoder expects different input shape
        decoded_images = self.decoder(decoded)
        
        
        # Now we should concat the slices to form the final 3D volume [batch, dh, dh, dh]
        decoded_images = decoded_images.view(batch_size, num_slices, self.dh, self.dh)
        volumes = []

        for i in range(decoded_images.shape[0]):  # Iterate over the batch dimension
            # Stack the 2D slices along a new dimension to form a 3D volume
            volume = torch.stack([decoded_images[i, j] for j in range(decoded_images.shape[1])], dim=0)
            # Now, volume is of shape [dh, dh, dh], forming a 3D volume for the ith item in the batch

            # Add a channel dimension
            volume = volume.unsqueeze(0)  # Now, volume is of shape [1, dh, dh, dh]

            volumes.append(volume)

        # Convert the list of volumes into a batched tensor
        volumes_tensor = torch.cat(volumes, dim=0).unsqueeze(1)  # Concatenates along the batch dimension
        return volumes_tensor
