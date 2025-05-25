import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_loss(x, x_reconstructed, mu, logvar):

    loss_reconstruction = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')

    # KL divergence (closed-form)
    loss_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = loss_reconstruction + loss_KL
    return loss, loss_reconstruction, loss_KL
