import torch
import random
from config import *
from dataset import setup_datasets_and_dataloaders
from loss import compute_loss
from train_utils import EarlyStopping, setup_commet_loger
from dataset import MidiDataset

def train(model, dataset_dir, experiment_name=None, verbose=True, model_save_path = "./saved_models/lofi-model.pth", weights_pth=None, early_stopping=True, previous_epoch_count=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if weights_pth:
        print("\n---WEIGHTS LOADING---")
        print(f"Loading weights from: {weights_pth}")
        weights = torch.load(weights_pth)
        model.load_state_dict(weights)
        model.to(device)
        print("Succesfully loaded weights.\n")
        print(f"Starting from epoch {previous_epoch_count}")
        print('_' * 60, "\n")

    train_dataloader, val_dataloader = setup_datasets_and_dataloaders(dataset_dir)
    
    if early_stopping:
        early_stopper = EarlyStopping(patience=30, path="checkpoints/best_model.pt", verbose=True)
    if experiment_name:
        experiment = setup_commet_loger(experiment_name)

    # Initialize teacher forcing ratio from config
    teacher_forcing_ratio = TEACHER_FORCING_RATIO

    print("\n=================")
    print("STARTING TRAINING")
    print("=================\n")

    print(f"Using {device} device\n")
    print(f"The datset has {len(train_dataloader)} batches\n")
    for epoch in range(NUM_EPOCHS):

        # --- KL WARM-UP AND ANNEALING SCHEDULE ---
        if epoch <= KLD_WARMUP_EPOCHS:
            kld_weight = 0.02
        else:
            kld_anneal_epochs = 100
            current_anneal_epoch = epoch - KLD_WARMUP_EPOCHS
            kld_weight = KLD_MAX_WEIGHT * (current_anneal_epoch / kld_anneal_epochs)
            kld_weight = min(kld_weight, KLD_MAX_WEIGHT)

        if experiment_name:
            experiment.log_metric("kld_weight", kld_weight, step=epoch)

        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_loss_reconstruction, train_loss_KL = 0, 0, 0

        print(f'Epoch [{epoch + 1 + previous_epoch_count}/{NUM_EPOCHS}]')
        for batch_idx, (sequences, lengths) in enumerate(train_dataloader):
            sequences = sequences.to(device)

            # Pass the current teacher_forcing_ratio to the model's forward method
            reconstructed_logits, mean, logvar = model(sequences, lengths, teacher_forcing_ratio)

            loss, loss_reconstruction, loss_KL = compute_loss(reconstructed_logits, sequences, mean, logvar, kld_weight=kld_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_loss_reconstruction += loss_reconstruction.item()
            train_loss_KL += loss_KL.item()

            if verbose and batch_idx > 0 and batch_idx % 100 == 0:
                avg_train_loss = train_loss / (batch_idx + 1)
                avg_train_loss_recon = train_loss_reconstruction / (batch_idx + 1)
                avg_train_loss_KL = train_loss_KL / (batch_idx + 1)

                print(f'\tBatch index: {batch_idx+1}/{len(train_dataloader)}')
                print(f'\tCurrent training Loss: {avg_train_loss:.4f}')
                print(f'\tCurrent training Reconstruction Loss: {avg_train_loss_recon:.4f}')
                print(f'\tCurrent training KL Loss: {avg_train_loss_KL:.4f}')
                print(f"\t\tKL weight: {kld_weight:.4f}")

                if experiment_name:
                    experiment.log_metric("batch_train_loss", avg_train_loss, step=epoch * len(train_dataloader) + batch_idx)
                    experiment.log_metric("batch_train_loss_reconstruction", avg_train_loss_recon, step=epoch * len(train_dataloader) + batch_idx)
                    experiment.log_metric("batch_train_loss_KL", avg_train_loss_KL, step=epoch * len(train_dataloader) + batch_idx)
                        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_loss_reconstruction, val_loss_KL = 0, 0, 0
        print("Validating:")
        with torch.no_grad():
            for batch_idx, (sequences, lengths) in enumerate(val_dataloader):
                sequences = sequences.to(device)

                # During validation, turn off teacher forcing (ratio=0.0) to get a true measure of performance
                reconstructed_logits, mean, logvar = model(sequences, lengths, 0.0)

                loss, loss_reconstruction, loss_KL = compute_loss(reconstructed_logits, sequences, mean, logvar, kld_weight=kld_weight)

                val_loss += loss.item()
                val_loss_reconstruction += loss_reconstruction.item()
                val_loss_KL += loss_KL.item()

        val_epoch_loss = val_loss / len(val_dataloader) 
        val_epoch_reconstruction_loss = val_loss_reconstruction / len(val_dataloader)
        val_epoch_KL_loss = val_loss_KL / len(val_dataloader)
        
        if verbose:
            print(f'Validation Reconstruction Loss: {val_epoch_reconstruction_loss:.4f}')
            print(f'Validation KL Loss: {val_epoch_KL_loss:.4f}')
            print(f'Validation Loss: {val_epoch_loss:.4f}')
            print(f"\t\tKL weight: {kld_weight:.4f}")
            print(f"\t\tTeacher forcing ratio: {teacher_forcing_ratio:.4f}")
            print('_' * 60, "\n")

        if experiment_name:
            experiment.log_metric("val_loss", val_epoch_loss, step=epoch)
            experiment.log_metric("epoch_val_loss_reconstruction", val_epoch_reconstruction_loss, step=epoch)
            experiment.log_metric("epoch_val_loss_KL", val_epoch_KL_loss, step=epoch)
            experiment.log_metric("teacher_forcing_ratio", teacher_forcing_ratio, step=epoch)

        torch.save(model.state_dict(), f"./saved_models/progress/lofi-model_epoch{epoch+1+previous_epoch_count}.pth")

        # Visualize some random samples reconstructed by the model
        if verbose and epoch % 10 == 0:
            random_num = torch.randint(0, len(train_dataloader.dataset), (1,)).item()
            random_tensor, random_length = train_dataloader.dataset[random_num]
            print(f"\nEpoch {epoch+1}. Visualizing random sample {random_num} from the training dataset:")
            MidiDataset.visualize_midi(random_tensor)
            print("Reconstructed sample:")
            model.reconstruct(random_tensor, random_length)
            print('_' * 60, "\n")

        if early_stopping and epoch > 40:
            early_stopper(val_epoch_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

        # Gently decay teacher forcing ratio for the next epoch
        if teacher_forcing_ratio > 0.001: # Don't let it decay to zero completely
            teacher_forcing_ratio *= 0.995 

    torch.save(model.state_dict(), model_save_path)
    print("TRAINING FINISHED")
    print("===================")
    print(f"Best model was saved to: {model_save_path}")

    if experiment_name:
        experiment.end()
