import torch
import torch.nn.functional as F
import math

def compute_improved_loss(recon_logits, x, mean, logvar, kld_weight=1.0, 
                         beta=1.0, focal_alpha=0.25, focal_gamma=2.0):
    """
    Improved VAE loss with several enhancements:
    1. Focal loss for handling class imbalance in sparse data
    2. β-VAE for better disentanglement
    3. Free bits for preventing posterior collapse
    4. Spectral regularization for smoother outputs
    """
    batch_size = x.size(0)
    
    # === RECONSTRUCTION LOSS with Focal Loss ===
    # Standard BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(
        recon_logits, x, reduction='none'
    )
    
    # Focal loss modification to handle sparse drum patterns
    probs = torch.sigmoid(recon_logits)
    pt = x * probs + (1 - x) * (1 - probs)  # Probability of true class
    focal_weight = focal_alpha * (1 - pt) ** focal_gamma
    focal_loss = focal_weight * bce_loss
    
    recon_loss = focal_loss.sum() / batch_size
    
    # === KL DIVERGENCE with Free Bits ===
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    
    # Free bits: prevent posterior collapse by ensuring minimum KL per dimension
    free_bits = 0.5  # Allow some dimensions to collapse
    kl_div_per_dim = kl_div / mean.size(1)
    kl_div_clamped = torch.clamp(kl_div_per_dim, min=free_bits) * mean.size(1)
    kl_div = kl_div_clamped.mean()
    
    # === SPECTRAL REGULARIZATION ===
    # Encourage smooth temporal transitions
    if recon_logits.size(2) > 1:  # If sequence length > 1
        temporal_diff = torch.diff(recon_logits, dim=2)
        spectral_loss = temporal_diff.pow(2).mean()
    else:
        spectral_loss = 0.0
    
    # === TOTAL LOSS ===
    total_loss = (recon_loss + 
                  beta * kld_weight * kl_div + 
                  0.01 * spectral_loss)  # Small spectral weight
    
    return total_loss, recon_loss, kl_div

def get_kl_weight_schedule(epoch, max_epochs, schedule_type="cosine", max_weight=1.0, warmup_epochs=50):
    """
    Different KL weight schedules for better training dynamics
    """
    if epoch < warmup_epochs:
        return 0.0
    
    progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
    progress = min(progress, 1.0)
    
    if schedule_type == "linear":
        return max_weight * progress
    elif schedule_type == "cosine":
        return max_weight * (1 - math.cos(progress * math.pi)) / 2
    elif schedule_type == "exponential":
        return max_weight * (1 - math.exp(-5 * progress))
    else:
        return max_weight * progress

def adaptive_teacher_forcing_ratio(epoch, val_loss_history, initial_ratio=1.0, 
                                  min_ratio=0.1, decay_rate=0.995):
    """
    Adaptive teacher forcing based on validation performance
    """
    base_ratio = max(initial_ratio * (decay_rate ** epoch), min_ratio)
    
    # If validation loss is increasing, increase teacher forcing
    if len(val_loss_history) >= 3:
        recent_trend = val_loss_history[-1] - val_loss_history[-3]
        if recent_trend > 0:  # Loss increasing
            adaptation_factor = 1.1
        else:  # Loss decreasing
            adaptation_factor = 0.9
        base_ratio *= adaptation_factor
    
    return min(max(base_ratio, min_ratio), 1.0)