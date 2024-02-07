import torch.nn as nn

class LRCN(nn.Module):
    def __init__(self):
        super(LRCN, self).__init__()
        self.lstm = nn.LSTM(input_size=... , hidden_size=..., num_layers=..., batch_first=True)
        self.fc = nn.Linear(..., output_size)

    def forward(self, x):
        # x is a batch of 2D slices
        # Process each slice through LSTM
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
