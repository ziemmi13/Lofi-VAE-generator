# in loss.py
import torch
import torch.nn.functional as F

def compute_loss(recon_x, x, mean, logvar, kld_weight=1.0):
    # --- CHANGE THIS LINE ---
    # recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence remains the same
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # The rest of the logic can stay the same
    total_loss = recon_loss + (kld_weight * kl_div)
    
    total_loss /= x.size(0)
    recon_loss /= x.size(0)
    kl_div /= x.size(0)
    
    return total_loss, recon_loss, kl_div