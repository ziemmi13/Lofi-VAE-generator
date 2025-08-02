# model.py
import torch
import torch.nn as nn
from config import *

class Encoder(nn.Module):
    def __init__(self, hidden_dim, z_dim, n_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # The input to the LSTM will be a flattened sequence of pitches for each time step.
        # Input features will be NUM_PITCHES * NUM_INSTRUMENTS
        self.lstm = nn.LSTM(
            NUM_PITCHES * NUM_INSTRUMENTS,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True # Using a bidirectional LSTM is often beneficial
        )
        
        # The bidirectional LSTM output is 2 * hidden_dim
        self.fc_mean = nn.Linear(2 * hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, z_dim)

    def forward(self, x, lengths):
        # x shape: (batch_size, num_instruments, num_pitches, max_len)
        
        # Reshape and permute for LSTM input
        # We want (batch_size, seq_len, features)
        x = x.permute(0, 3, 1, 2) 
        # New shape: (batch_size, max_len, num_instruments, num_pitches)
        x = x.reshape(x.size(0), x.size(1), -1) 
        # New shape: (batch_size, max_len, num_instruments * num_pitches)

        # Pack padded sequence to handle variable lengths
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM forward pass
        _, (hidden, _) = self.lstm(packed_x)
        
        # Concatenate the hidden states from both directions of the last layer
        # Hidden shape: (n_layers * 2, batch_size, hidden_dim)
        # We take the last layer's hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Get mean and log variance
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, n_layers):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # LSTM for decoding
        self.lstm = nn.LSTM(
            z_dim,
            hidden_dim,
            n_layers,
            batch_first=True
        )
        
        # Fully connected layer to reconstruct the piano roll
        self.fc = nn.Linear(hidden_dim, NUM_PITCHES * NUM_INSTRUMENTS)

    def forward(self, z, max_len):
        # z shape: (batch_size, z_dim)
        
        # Repeat z for each time step up to max_len
        # This provides the latent vector as input at each decoding step
        z_repeated = z.unsqueeze(1).repeat(1, max_len, 1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(z_repeated)
        
        # Fully connected layer to get output features
        # lstm_out shape: (batch_size, max_len, hidden_dim)
        output = self.fc(lstm_out)
        
        # Reshape to piano roll format and apply sigmoid
        # Sigmoid is used because input pixels (velocities) are normalized between 0 and 1
        output = torch.sigmoid(output)
        output = output.view(output.size(0), output.size(1), NUM_INSTRUMENTS, NUM_PITCHES)
        # Permute to match input shape: (batch_size, num_instruments, num_pitches, max_len)
        output = output.permute(0, 2, 3, 1)
        
        return output

class LSTMVae(nn.Module):
    def __init__(self, hidden_dim=LATENT_DIM, z_dim=LATENT_DIM, n_layers=LSTM_LAYERS):
        super(LSTMVae, self).__init__()
        self.encoder = Encoder(hidden_dim, z_dim, n_layers)
        self.decoder = Decoder(z_dim, hidden_dim, n_layers)

    def reparameterize(self, mean, logvar):
        # Standard deviation
        std = torch.exp(0.5 * logvar)
        # Random noise
        eps = torch.randn_like(std)
        # Sampling
        return mean + eps * std

    def forward(self, x, lengths):
        mean, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z, x.size(3)) # x.size(3) is max_len
        return reconstruction, mean, logvar

