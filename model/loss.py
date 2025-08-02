import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_loss(x, recon_x, mean, logvar, lengths):
    # Create a mask to exclude padded parts from the loss calculation
    max_len = x.size(3)
    mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(1).unsqueeze(1).expand_as(x)

    # 1. Reconstruction Loss (Binary Cross-Entropy)
    # We only care about the loss for the actual sequence, not the padding
    recon_loss = nn.functional.binary_cross_entropy(recon_x[mask], x[mask], reduction='sum')

    # 2. KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # The total loss is the sum, normalized by batch size
    total_loss = (recon_loss + kl_div) / x.size(0)
    return total_loss, recon_loss, kl_div   
