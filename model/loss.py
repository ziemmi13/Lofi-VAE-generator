import torch
import torch.nn.functional as F

def compute_loss(x, x_reconstructed, distribution, z, loss_type):
    # reconstruction loss
    if loss_type == "BCE":
        # loss_reconstruction = F.binary_cross_entropy(x_reconstructed, x, reduction='none').sum(-1).mean()
        loss_reconstruction = F.binary_cross_entropy(x_reconstructed, x, reduction='sum') / x.size(0)

    elif loss_type == "MSE":
        loss_reconstruction = F.mse_loss(x_reconstructed, x, reduction='none').sum(-1).mean()

    # KL Divergence
    # Normal distribution N(0,1) with size like distribution from encode()
    std_normal = torch.distributions.MultivariateNormal(
        torch.zeros_like(z, device=z.device),
        scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
    )
    loss_KL = torch.distributions.kl.kl_divergence(distribution, std_normal).mean()

    loss = loss_reconstruction + loss_KL

    
    return loss, loss_reconstruction, loss_KL
