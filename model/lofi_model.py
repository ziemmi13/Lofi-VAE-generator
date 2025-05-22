import numpy as np
import torch
from torch import nn
from config import *

class LofiModel(nn.Module):
    def __init__(self, device):
        super(LofiModel, self).__init__()
        self.device = device
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.softplus = nn.Softplus()

    def encode(self, x, eps=1e-8):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability - avoid /0 or log(0)
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=-1) # torch.chunk - Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.

        # logvar can be <0 we dont want that 
        # self.softplus(logvar) works like log(1 + exp(logvar)) so that scale is always > 0
        # Softmax prevents vanishing gradients 
        std = self.softplus(logvar) + eps
        # std = torch.exp(0.5 * logvar)

        # changes scale size from [batch_size, latent_dim] -> [batch_size, latent_dim, latent_dim] (makes it a diagonal matrix)
        # scale_tril = torch.diag_embed(scale)
        
        scale_tril = torch.diag_embed(std)
        # Multidimentional Normal Distribution with avg mu and covariance matrix scale_tril
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def decode(self, z):

        return self.decoder(z)
    
    def reparameterize(self, distribution):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            distribution (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        # rsample (reparametrized sample) is just z = mu + scale * ε 
        # keeps the computation graph intact and enables backpropagation
        # return dist.rsample()
        # rsample (reparametrized sample) is just z = mu + scale * ε 
        # keeps the computation graph intact and enables backpropagation
        return distribution.rsample()
        

    def forward(self, x):
        """
        Performs a forward pass of VAE  
        Args:
            x (torch.Tensor): Input MIDI

        Returns:
            distribution (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
            z (torch.Tensor): Latent vector 
            reconstructed_x (torch.Tensor): Reconstructed input 

        """
        distribution = self.encode(x)
        z = self.reparameterize(distribution)
        reconstructed_x = self.decode(z)

        return distribution, z, reconstructed_x

class Encoder(nn.Module):
    def __init__(self, input_dim=PIANOROLL_RANGE, hidden_dim=256, latent_dim=LATENT_DIM, dropout=0.1):
        super(Encoder, self).__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout)

        self.layer_norm = nn.LayerNorm(hidden_dim*2) # 2 * for mean and variance
        self.dropout = nn.Dropout(dropout)

        # One linear for mu and logvar
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2), # 2 * for mean and variance
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, latent_dim*2) # 2 * for mean and variance
         ) 

    def forward(self, x):
        x = self.input_projection(x)
        
        lstm_out, (h_n, c_n) = self.lstm(x) # lstm_out - output features, h_n - hiden state, c_n - final cell state  
        
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, hidden_dim * 2)

        h_n = self.dropout(self.layer_norm(h_n))

        x = self.fc(h_n)

        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=256, output_dim=PIANOROLL_RANGE, seq_len=MIDI_LEN, dropout=0.1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Linear projection
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim*2)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim*2)

        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Output layer to project LSTM output to piano roll features (pitches)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        batch_size = z.size(0)

        # Prepare initial hidden and cell states for LSTM
        h_0_flat = self.latent_to_hidden(z) # (batch_size, hidden_dim * num_layers)
        c_0_flat = self.latent_to_cell(z) # (batch_size, hidden_dim * num_layers)

        # Reshape to (num_layers, batch_size, hidden_dim) for LSTM
        h_0 = h_0_flat.view(batch_size, 2, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = c_0_flat.view(batch_size, 2, self.hidden_dim).permute(1, 0, 2).contiguous()


        lstm_input = torch.zeros(batch_size, MIDI_LEN, self.hidden_dim, device=z.device)
        lstm_out, _ = self.lstm(lstm_input, (h_0, c_0))

        reconstruction = self.output_projection(lstm_out)
        reconstruction = self.sigmoid(reconstruction)

        # TODO
        # Use teacher forcing

        return reconstruction

        
