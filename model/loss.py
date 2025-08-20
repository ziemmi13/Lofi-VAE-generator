import torch
import torch.nn.functional as F

def compute_loss(recon_logits, x, mean, logvar, kld_weight=1.0):
    """
    Computes the VAE loss using Binary Cross-Entropy with Logits.
    This is more numerically stable than a separate Sigmoid followed by BCE.
    """

    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, x, reduction='sum')

    # KL-Divergence remains the same.
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Combine the two terms.
    total_loss = recon_loss + (kld_weight * kl_div)
    
    # Normalize by batch size for consistent logging.
    total_loss /= x.size(0)
    recon_loss /= x.size(0)
    kl_div /= x.size(0)
    
    return total_loss, recon_loss, kl_div