import numpy as np
import torch
from torch import nn

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
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1) # torch.chunk - Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.

        # logvar can be <0 we dont want that 
        # self.softplus(logvar) works like log(1 + exp(logvar)) so that scale is always > 0
        # Softmax prevents vanishing gradients 
        scale = self.softplus(logvar) + eps

        # changes scale size from [batch_size, latent_dim] -> [batch_size, latent_dim, latent_dim] (makes it a diagonal matrix)
        scale_tril = torch.diag_embed(scale)
        
        # Multidimentional Normal Distribution with avg mu and covariance matrix scale_tril
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def decode(self, z):

        return self.decoder(z)
    
    def reparametrize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        # rsample (reparametrized sample) is just z = mu + scale * ε 
        # keeps the computation graph intact and enables backpropagation
        return dist.rsample()

    def forward(self, x):
        """
        Performs a forward pass of VAE  
        Args:

        Returns:
            distribution ():
            z (): 
            reconstructed_x (): 

        """
        distribution = self.encode(x)
        z = self.reparametrize(distribution)
        reconstructed_x = self.decode(z)

        return distribution, z, reconstructed_x

class Encoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=64):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim, latent_dim * 2) # 2 * for mean and variance

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x) # x - output features, h_n - hiden state, c_n - final cell state  
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, hidden_dim * 2)
        x = self.fc1(h_n)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=88, seq_len=388):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = self.fc1(z)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        x, (h_n, c_n) = self.lstm(x)
        x = self.fc2(x)
        # TODO
        # Use teacher forcing
        return x
