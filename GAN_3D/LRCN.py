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
            nn.Conv2d(num_slices, 64, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_slices, kernel_size=(5,5), stride=(2,2), padding=(2,2), output_padding=(1,1)),
            nn.Tanh()
        )
        
    def forward(self, x):
        # X is a sequence of 2D images of size (dl x dl), of length num_slices.. thus x.shape = [batch_size, num_slices, dl, dl]
        batch_size, c, h, w = x.size()
        # Reshape for encoder [batch_size, num_slices, dl, dl] -> [batch_size, channels, num_slices, dl, dl] (making the sequence a sequence of 3D images of size (num_slices x dl x dl))
        x = x.view(batch_size, 1, c, h, w)
        print("Input shape: ", x.shape)
        x = self.encoder(x)
        print("Encoder out shape: ", x.shape)
        # x is now a 1D feature vectors of size (lstm_hidden_size)
        x = x.reshape(batch_size, self.lstm_hidden_size)
        print("Reshaped input shape: ", x.shape)
        lstm_out, _ = self.lstm(x)
        #lstm_out = lstm_out.reshape(batch_size, 1, -1, 1)
        lstm_out = self.decoder_fc(lstm_out)
        lstm_out = lstm_out.view(batch_size, self.num_slices, self.dh, self.dh)
        print("LSTM out shape: ", lstm_out.shape)
        decoded_images = self.decoder(lstm_out)
        print("Decoded images shape: ", decoded_images.shape)

        return decoded_images
    
    