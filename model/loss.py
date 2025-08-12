import torch
import torch.nn.functional as F

def compute_loss(recon_x, x, mean, logvar, kld_weight=1.0):
    """
    Computes the VAE loss, which is a sum of the reconstruction loss and the KL-divergence.
    """
    # Reconstruction Loss: How well the model reconstructs the input.
    # We use Mean Squared Error, which is a good choice for this type of continuous data.
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL-Divergence: A regularizer that forces the latent space to be smooth and continuous.
    # It measures the difference between the learned latent distribution and a standard normal distribution.
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Combine the two terms. The kld_weight is controlled by the training loop (annealing).
    # total_loss = recon_loss + (kld_weight * kl_div)
    total_loss = recon_loss + (kld_weight * kl_div)
    
    # Normalize by batch size for consistent logging across different batch sizes.
    total_loss /= x.size(0)
    recon_loss /= x.size(0)
    kl_div /= x.size(0)
    
    return total_loss, recon_loss, kl_div
