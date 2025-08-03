# model.py
import torch
import torch.nn as nn
from config import *
from dataset import MidiDataset

class Encoder(nn.Module):
    def __init__(self, hidden_dim, z_dim, n_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # The input to the LSTM will be a flattened sequence of pitches for each time step.
        # Input features will be NUM_PITCHES
        self.lstm = nn.LSTM(
            NUM_PITCHES,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True # Using a bidirectional LSTM is often beneficial
        )
        
        # The bidirectional LSTM output is 2 * hidden_dim
        self.fc_mean = nn.Linear(2 * hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, z_dim)

    def forward(self, x, lengths):
        # x shape: (batch_size, num_pitches, max_len)
        
        # Reshape and permute for LSTM input
        # We want (batch_size, seq_len, features)
        x = x.permute(0, 2, 1) 
        # New shape: (batch_size, max_len, num_pitches)

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
            NUM_PITCHES,
            hidden_dim,
            n_layers,
            batch_first=True
        )
        
        # Fully connected layer to reconstruct the piano roll
        self.fc = nn.Linear(hidden_dim, NUM_PITCHES)

    def forward(self, x, hidden, cell):
        # x is now the input sequence for teacher forcing
        # hidden and cell are the initial states derived from z
        # x shape: (batch_size, max_len, num_pitches)
        
        lstm_out, _ = self.lstm(x, (hidden, cell))
        output = self.fc(lstm_out)
        
        # We will apply sigmoid in the main model class
        # Permute to match target shape
        output = output.permute(0, 2, 1)
        return output

class LofiModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, n_layers=LSTM_LAYERS):
        super(LofiModel, self).__init__()
        self.encoder = Encoder(hidden_dim, latent_dim, n_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, n_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc_latent_to_hidden = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_latent_to_cell = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

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

        # --- MAP Z TO DECODER'S INITIAL STATE ---
        hidden_init_flat = self.fc_latent_to_hidden(z)
        cell_init_flat = self.fc_latent_to_cell(z)

        batch_size = x.size(0)
        # Reshape to (n_layers, batch_size, hidden_dim)
        decoder_hidden_init = hidden_init_flat.view(self.n_layers, batch_size, self.hidden_dim)
        decoder_cell_init = cell_init_flat.view(self.n_layers, batch_size, self.hidden_dim)

        # --- DECODE with TEACHER FORCING ---
        # The decoder's input should be the original sequence permuted for LSTM
        decoder_input = x.permute(0, 2, 1) # Shape: (batch_size, max_len, num_pitches)
        
        reconstruction = self.decoder(decoder_input, decoder_hidden_init, decoder_cell_init)
        
        # Apply final activation function here
        return reconstruction, mean, logvar
    
    def generate(self, max_len):
        # Generate a random latent vector
        z = torch.randn(1, LATENT_DIM).to(next(self.parameters()).device)
        # Decode the latent vector to generate a sequence
        generated_sequence = self.decoder(z, max_len)
        return generated_sequence
    
    def reconstruct(self, x, lengths):
        self.eval()

        lengths_tensor = torch.tensor([lengths], dtype=torch.long)
        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructed_x, _, _ = self(x, lengths_tensor)
        
        reconstructed_x = reconstructed_x.squeeze(0)  # Remove batch dimension
        # Threshold small values to zero
        # reconstructed_x[reconstructed_x < 0.05] = 0.0
        reconstructed_x = reconstructed_x.cpu()
        MidiDataset.visualize_midi(reconstructed_x)

