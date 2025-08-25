import torch
import random
from config import *
from dataset import setup_datasets_and_dataloaders
from loss import compute_improved_loss, get_kl_weight_schedule, adaptive_teacher_forcing_ratio
from train_utils import EarlyStopping, setup_commet_loger
from dataset import MidiDataset
import math

def train(model, dataset_dir, experiment_name=None, verbose=True, model_save_path = "./saved_models/lofi-model.pth", weights_pth=None, early_stopping=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Use different optimizers for different components if desired
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.1
    )

    if weights_pth:
        print("\n---WEIGHTS LOADING---")
        print(f"Loading weights from: {weights_pth}")
        weights = torch.load(weights_pth, map_location=device)
        model.load_state_dict(weights)
        model.to(device)
        print("Successfully loaded weights.\n")
        print('_' * 60, "\n")

    train_dataloader, val_dataloader = setup_datasets_and_dataloaders(dataset_dir)
    
    if early_stopping:
        early_stopper = EarlyStopping(patience=100, path="checkpoints/best_model.pt", verbose=True)  # Increased patience
    if experiment_name:
        experiment = setup_commet_loger(experiment_name)

    # Initialize teacher forcing ratio from config
    teacher_forcing_ratio = TEACHER_FORCING_RATIO
    val_loss_history = []  # For adaptive teacher forcing

    print("\n=================")
    print("STARTING TRAINING")
    print("=================\n")

    print(f"Using {device} device\n")
    print(f"The dataset has {len(train_dataloader)} batches\n")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    for epoch in range(NUM_EPOCHS):

        # --- IMPROVED KL WARM-UP AND ANNEALING SCHEDULE ---
        kld_weight = get_kl_weight_schedule(
            epoch, NUM_EPOCHS, 
            schedule_type=KL_WEIGHT_SCHEDULE, 
            max_weight=KLD_MAX_WEIGHT, 
            warmup_epochs=KLD_WARMUP_EPOCHS
        )

        # --- ADAPTIVE TEACHER FORCING ---
        teacher_forcing_ratio = adaptive_teacher_forcing_ratio(
            epoch, val_loss_history, 
            initial_ratio=TEACHER_FORCING_RATIO,
            min_ratio=TEACHER_FORCING_MIN,
            decay_rate=TEACHER_FORCING_DECAY)


        # β-VAE scheduling (optional)
        # Gradually increase β for better disentanglement
        beta = min(BETA_VAE_BETA * (epoch / 100), BETA_VAE_BETA)


        # Logging
        if experiment_name:
            experiment.log_metric("kld_weight", kld_weight, step=epoch)
            experiment.log_metric("teacher_forcing_ratio", teacher_forcing_ratio, step=epoch)
            experiment.log_metric("beta_vae_beta", beta, step=epoch)
            experiment.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_loss_reconstruction, train_loss_KL = 0, 0, 0
        num_batches = len(train_dataloader)

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] (LR: {optimizer.param_groups[0]["lr"]:.2e})')
        for batch_idx, (sequences, lengths) in enumerate(train_dataloader):
            sequences = sequences.to(device)

            # Forward pass with improved model
            reconstructed_logits, mean, logvar = model(sequences, lengths, teacher_forcing_ratio, beta)


            # Use improved loss function if available
            loss, loss_reconstruction, loss_KL = compute_improved_loss(
                reconstructed_logits, sequences, mean, logvar, 
                kld_weight=kld_weight, beta=beta
            )


            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            
            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            
            optimizer.step()

            train_loss += loss.item()
            train_loss_reconstruction += loss_reconstruction.item()
            train_loss_KL += loss_KL.item()

            # Enhanced logging with more metrics
            if verbose and batch_idx > 0 and batch_idx % 50 == 0:
                avg_train_loss = train_loss / (batch_idx + 1)
                avg_train_loss_recon = train_loss_reconstruction / (batch_idx + 1)
                avg_train_loss_KL = train_loss_KL / (batch_idx + 1)

                # Calculate additional metrics
                kl_per_dim = avg_train_loss_KL / LATENT_DIM
                
                print(f'\tBatch index: {batch_idx+1}/{len(train_dataloader)}')
                print(f'\tCurrent training Loss: {avg_train_loss:.4f}')
                print(f'\tCurrent training Reconstruction Loss: {avg_train_loss_recon:.4f}')
                print(f'\tCurrent training KL Loss: {avg_train_loss_KL:.4f} (per dim: {kl_per_dim:.4f})')
                print(f"\t\tKL weight: {kld_weight:.4f}")
                print(f"\t\tβ-VAE β: {beta:.4f}")

                if experiment_name:
                    step = epoch * len(train_dataloader) + batch_idx
                    experiment.log_metric("batch_train_loss", avg_train_loss, step=step)
                    experiment.log_metric("batch_train_loss_reconstruction", avg_train_loss_recon, step=step)
                    experiment.log_metric("batch_train_loss_KL", avg_train_loss_KL, step=step)
                    experiment.log_metric("batch_kl_per_dim", kl_per_dim, step=step)
                        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_loss_reconstruction, val_loss_KL = 0, 0, 0
        print("Validating:")
        with torch.no_grad():
            for batch_idx, (sequences, lengths) in enumerate(val_dataloader):
                sequences = sequences.to(device)

                # During validation, turn off teacher forcing (ratio=0.0) to get a true measure of performance
                reconstructed_logits, mean, logvar = model(sequences, lengths, 0.0, beta)

                loss, loss_reconstruction, loss_KL = compute_improved_loss(
                    reconstructed_logits, sequences, mean, logvar, 
                    kld_weight=kld_weight, beta=beta
                )

                val_loss += loss.item()
                val_loss_reconstruction += loss_reconstruction.item()
                val_loss_KL += loss_KL.item()

        val_epoch_loss = val_loss / len(val_dataloader) 
        val_epoch_reconstruction_loss = val_loss_reconstruction / len(val_dataloader)
        val_epoch_KL_loss = val_loss_KL / len(val_dataloader)
        
        # Store for adaptive teacher forcing
        val_loss_history.append(val_epoch_loss)
        if len(val_loss_history) > 10:  # Keep only recent history
            val_loss_history.pop(0)
        
        # Enhanced validation logging
        val_kl_per_dim = val_epoch_KL_loss / LATENT_DIM 
        
        if verbose:
            print(f'Validation Reconstruction Loss: {val_epoch_reconstruction_loss:.4f}')
            print(f'Validation KL Loss: {val_epoch_KL_loss:.4f} (per dim: {val_kl_per_dim:.4f})')
            print(f'Validation Loss: {val_epoch_loss:.4f}')
            print(f"\t\tKL weight: {kld_weight:.4f}")
            print(f"\t\tβ-VAE β: {beta:.4f}")
            print(f"\t\tTeacher forcing ratio: {teacher_forcing_ratio:.4f}")
            print('_' * 60, "\n")

        if experiment_name:
            experiment.log_metric("val_loss", val_epoch_loss, step=epoch)
            experiment.log_metric("epoch_val_loss_reconstruction", val_epoch_reconstruction_loss, step=epoch)
            experiment.log_metric("epoch_val_loss_KL", val_epoch_KL_loss, step=epoch)
            experiment.log_metric("val_kl_per_dim", val_kl_per_dim, step=epoch)

        # Save progress with more frequent checkpoints for good models
        torch.save(model.state_dict(), f"./saved_models/progress/lofi-model_epoch{epoch+1}.pth")
        
        # Save best model based on reconstruction loss (more relevant for generation quality)
        if epoch == 0 or val_epoch_reconstruction_loss < best_recon_loss:
            best_recon_loss = val_epoch_reconstruction_loss
            torch.save(model.state_dict(), "./saved_models/best_reconstruction_model.pth")

        # Visualize samples with enhanced monitoring
        if verbose and epoch % 50 == 0:
            random_num = torch.randint(0, len(train_dataloader.dataset), (1,)).item()
            random_tensor, random_length = train_dataloader.dataset[random_num]
            print(f"\nEpoch {epoch+1}. Visualizing random sample {random_num} from the training dataset:")
            MidiDataset.visualize_midi(random_tensor)

            print("Reconstructed sample:")
            model.reconstruct(random_tensor, random_length, visualize=True)
            
            # Test generation
            print("Generated sample:")
            model.generate(num_samples=1, temperature=0.8, device=device, visualize=True, save_path=None)
            print('_' * 60, "\n")
           

        # Early stopping with improved criteria
        if early_stopping and epoch > 150:  # Start later to allow proper warmup
            early_stopper(val_epoch_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

        # Step the scheduler
        scheduler.step()
        
        # Initialize best_recon_loss for the first epoch
        if epoch == 0:
            best_recon_loss = val_epoch_reconstruction_loss

    torch.save(model.state_dict(), model_save_path)
    print("TRAINING FINISHED")
    print("===================")
    print(f"Best model was saved to: {model_save_path}")
    print(f"Best reconstruction model saved to: ./saved_models/best_reconstruction_model.pth")

    if experiment_name:
        experiment.end()