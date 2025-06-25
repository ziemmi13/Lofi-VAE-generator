import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_loss(x, x_reconstructed, mu, logvar, loss_type, beta=1.0):

    if loss_type == "BCE":
        loss_reconstruction = nn.BCEWithLogitsLoss(reduction='sum')(x_reconstructed, x)
    elif loss_type == "MSE":
        loss_reconstruction = F.mse_loss(x_reconstructed, x, reduction='sum') # TODO Check reduction='mean'

    # KL divergence (closed-form)
    loss_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = loss_reconstruction + (beta * loss_KL)
    return loss, loss_reconstruction, loss_KL
