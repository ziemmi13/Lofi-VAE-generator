import numpy as np
import torch
from torch import nn

class LofiModel(nn.Module):
    def __init__(self, device):
        super(LofiModel, self).__init__()
        self.device = device
        self.encoder = Encoder()
        self.decoder = Decoder()

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
        # Latent sample
        z = self.reparametrize(distribution)
        reconstructed_x = self.decode(z)

        return distribution, z, reconstructed_x




class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device

    def forward(self, x):
        pass

class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.device = device

    def forward(self, x):
        pass

