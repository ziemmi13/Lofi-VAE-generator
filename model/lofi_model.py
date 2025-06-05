import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import *

class LofiModel(nn.Module):
    def __init__(self, device):
        super(LofiModel, self).__init__()
        self.device = device
        # Decoder's input_dim for LSTM should match INPUT_DIM for teacher forcing
        self.encoder = Encoder(input_dim=INPUT_DIM, hidden_dim=LATENT_DIM, latent_dim=LATENT_DIM, num_layers=LSTM_LAYERS)
        self.decoder = Decoder(rnn_input_dim=INPUT_DIM, # For teacher forcing, LSTM input is from x (INPUT_DIM)
                               lstm_hidden_dim=LATENT_DIM,
                               z_latent_dim=LATENT_DIM,
                               output_feature_dim=INPUT_DIM,
                               num_layers=LSTM_LAYERS)

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            logvar (torch.Tensor): Log variance of the latent Gaussian.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        z = mu + eps * std
        return z
        
    def forward(self, x, lengths):
        """
        Performs a forward pass of VAE  
        Args:
            x (torch.Tensor): Input MIDI (batch_size, max_seq_len, input_dim).
            lengths (torch.Tensor): Original lengths of sequences in the batch.

        Returns:
            reconstructed_x (torch.Tensor): Reconstructed input (batch_size, max_seq_len, input_dim).
            mu (torch.Tensor): Latent mean (batch_size, latent_dim).
            logvar (torch.Tensor): Latent log variance (batch_size, latent_dim).
        """
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        # For teacher forcing, decoder uses original x as input sequence
        reconstructed_x = self.decoder(x, z, lengths)

        return reconstructed_x, mu, logvar

class Encoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=LATENT_DIM, latent_dim=LATENT_DIM, num_layers=LSTM_LAYERS, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim # LSTM hidden size
        self.num_layers = num_layers
        self.latent_dim = latent_dim # VAE latent z dimension

        self.lstm = nn.LSTM(input_size=input_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, # Corrected: use self.num_layers
                              batch_first=True,
                              dropout=dropout if self.num_layers > 1 else 0.0) # Corrected: use dropout param

        # Project LSTM's final hidden state to mean and logvar of latent distribution
        # Input: (batch, num_layers * hidden_dim) -> Output: (batch, latent_dim)
        self.hidden_to_mu = nn.Linear(self.hidden_dim * self.num_layers, self.latent_dim)
        self.hidden_to_logvar = nn.Linear(self.hidden_dim * self.num_layers, self.latent_dim)

    def forward(self, x, lengths):
        """
        Forward pass for the encoder.
        Args:
            x (torch.Tensor): Padded input sequences (batch_size, max_seq_len, input_dim).
            lengths (torch.Tensor): Original lengths of sequences in the batch.
        Returns:
            tuple: mu and logvar of the latent space (each shape: batch_size, latent_dim)
        """
        # Pack padded batch of sequences for the LSTM
        # Ensure lengths are on CPU and are integers/long for pack_padded_sequence
        packed_input = pack_padded_sequence(x, lengths.cpu().long(), batch_first=True, enforce_sorted=False)

        # Pass the packed input through the LSTM.
        # hidden and cell are the final states: (num_layers * num_directions, batch, hidden_size)
        _, (hidden, cell) = self.lstm(packed_input)

        # Flatten hidden layers: (num_layers, batch, hidden_dim) -> (batch, num_layers * hidden_dim)
        # LSTM's hidden is (D*num_layers, N, H_out) where D=1 for non-bidirectional
        # Permute to (N, D*num_layers, H_out) then view as (N, D*num_layers*H_out)
        hidden = hidden.permute(1, 0, 2).contiguous().view(x.size(0), -1)

        # mu and logvar
        mu = self.hidden_to_mu(hidden)
        logvar = self.hidden_to_logvar(hidden)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, rnn_input_dim=INPUT_DIM, lstm_hidden_dim=LATENT_DIM, 
                 z_latent_dim=LATENT_DIM, output_feature_dim=INPUT_DIM, 
                 num_layers=LSTM_LAYERS, dropout=0.1):
        super(Decoder, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.z_latent_dim = z_latent_dim
        self.output_feature_dim = output_feature_dim

        # Project latent vector z to initial hidden and cell states for the LSTM
        # Input: z_latent_dim -> Output: num_layers * lstm_hidden_dim
        self.latent_to_hidden = nn.Linear(self.z_latent_dim, self.num_layers * self.lstm_hidden_dim)
        self.latent_to_cell = nn.Linear(self.z_latent_dim, self.num_layers * self.lstm_hidden_dim)

        # LSTM:
        # input_size: dimension of input features at each time step (from x for teacher forcing)
        # hidden_size: dimension of LSTM hidden units
        self.lstm = nn.LSTM(input_size=rnn_input_dim, # Corrected: This is the dim of teacher-forced input x_t
                              hidden_size=self.lstm_hidden_dim,
                              num_layers=self.num_layers, # Corrected: use self.num_layers
                              batch_first=True,
                              dropout=dropout if self.num_layers > 1 else 0.0) # Corrected: use dropout param

        # Output layer: maps LSTM hidden state to output feature dimension
        self.fc_out = nn.Linear(self.lstm_hidden_dim, self.output_feature_dim)
        
        # If your output features are probabilities (e.g., for a piano roll),
        # you might want a sigmoid activation here or after the loss calculation.
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x_teacher_force, z, lengths):
        """
        Forward pass for the decoder.
        Args:
            x_teacher_force (torch.Tensor): Ground truth sequences for teacher forcing
                                           (batch_size, max_seq_len, rnn_input_dim).
            z (torch.Tensor): Latent vector (batch_size, z_latent_dim).
            lengths (torch.Tensor): Original lengths of sequences in the batch.
        Returns:
            torch.Tensor: Reconstructed sequences (batch_size, max_seq_len, output_feature_dim).
        """
        batch_size = z.size(0)
        max_len = x_teacher_force.shape[1]

        # Project latent vector z to initial hidden and cell states
        # Output shape: (batch_size, num_layers * lstm_hidden_dim)
        initial_hidden_flat = self.latent_to_hidden(z)
        initial_cell_flat = self.latent_to_cell(z)

        # Reshape to (num_layers, batch_size, lstm_hidden_dim) for LSTM
        h_0 = initial_hidden_flat.view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = initial_cell_flat.view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()

        # Prepare input sequence for LSTM (using teacher forcing with x_teacher_force)
        # x_teacher_force has shape (batch_size, max_seq_len, rnn_input_dim)
        # Ensure lengths are on CPU and are integers/long for pack_padded_sequence
        packed_input = pack_padded_sequence(x_teacher_force, lengths.cpu().long(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, _ = self.lstm(packed_input, (h_0, c_0))
        
        # Unpack the output sequence
        # output shape: (batch_size, max_seq_len, lstm_hidden_dim)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)
        
        # Pass LSTM outputs through the final fully connected layer
        # reconstructed_x shape: (batch_size, max_seq_len, output_feature_dim)
        reconstructed_x = self.fc_out(output)

        # if self.sigmoid:
        #     reconstructed_x = self.sigmoid(reconstructed_x)
            
        return reconstructed_x