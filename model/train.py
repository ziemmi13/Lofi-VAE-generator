import torch
from config import *
from dataset import setup_datasets_and_dataloaders
from loss import compute_loss
from train_utils import EarlyStopping

def train(model, dataset_dir, verbose=True, model_save_path = "./saved_models/lofi-model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_dataloader, val_dataloader = setup_datasets_and_dataloaders(dataset_dir)
    
    early_stopper = EarlyStopping(patience=5, path="checkpoints/best_model.pt")

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_loss_reconstruction = 0
        train_loss_KL = 0

        for batch_idx, (sequences, lengths) in enumerate(train_dataloader):
            sequences = sequences.to(device)

            optimizer.zero_grad()

            reconstructed_logits, mu, logvar = model(sequences, lengths)

            # Compute loss
            loss, loss_reconstruction, loss_KL = compute_loss(sequences, reconstructed_logits, mu, logvar)

            train_loss += loss.item()
            train_loss_reconstruction += loss_reconstruction.item()
            train_loss_KL += loss_KL.item()

            # Update network
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss = train_loss / len(train_dataloader)
        epoch_reconstruction_loss = train_loss_reconstruction / len(train_dataloader)
        epoch_KL = train_loss_KL / len(train_dataloader)
        # # Log epoch metrics
        # experiment.log_metric("epoch_train_loss", epoch_loss, step=epoch)
        # experiment.log_metric("epoch_train_loss_reconstruction", epoch_reconstruction_loss, step=epoch)
        # experiment.log_metric("epoch_train_loss_KL", epoch_KL, step=epoch)

        # Validation phase
        model.eval()
        val_loss = 0
        val_loss_reconstruction = 0
        val_loss_KL = 0
        with torch.no_grad():
            for batch_idx, (sequences, lengths) in enumerate(val_dataloader):
                sequences = sequences.to(device)

                reconstructed_logits, mu, logvar = model(sequences, lengths)

                # Compute loss
                loss, loss_reconstruction, loss_KL = compute_loss(sequences, reconstructed_logits, mu, logvar)

                val_loss += loss.item()
                val_loss_reconstruction += loss_reconstruction.item()
                val_loss_KL += loss_KL.item()

        val_epoch_loss = val_loss / len(val_dataloader) 
        val_epoch_reconstruction_loss = val_loss_reconstruction / len(val_dataloader)
        val_epoch_KL_loss = val_loss_KL / len(val_dataloader)
        
        if verbose:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
            print(f'Training Loss: {epoch_loss:.4f}')
            print(f'Reconstruction Loss: {val_epoch_reconstruction_loss:.4f}')
            print(f'KL Loss: {val_epoch_KL_loss:.4f}')
            print()
            print(f'Validation Loss: {val_epoch_loss:.4f}')
            print('-' * 60, "\n")

        # # Log validation metrics
        # experiment.log_metric("val_loss", val_epoch_loss, step=epoch)
        # experiment.log_metric("epoch_val_loss_reconstruction", val_epoch_reconstruction_loss, step=epoch)
        # experiment.log_metric("epoch_val_loss_KL", val_epoch_KL, step=epoch)

        # Early stopping


        # Saving the trained model
        early_stopper(val_epoch_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print("Finished training!")
    print(f"Saving the model to path: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

    print("Model saved!")




