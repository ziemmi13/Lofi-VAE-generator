import torch
from config import *
from dataset import setup_datasets_and_dataloaders
from loss import compute_loss
from train_utils import EarlyStopping, setup_commet_loger
from config import *
from dataset import MidiDataset
# from tqdm import tqdm

def train(model, dataset_dir, experiment_name, verbose=True, model_save_path = "./saved_models/lofi-model.pth", weights_pth=None, early_stopping=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Load weights if available
    if weights_pth:
        print("\n---WEIGHTS LOADING---")
        print(f"Loading weights from: {weights_pth}")
        weights = torch.load(weights_pth)
        model.load_state_dict(weights)
        model.to(device)
        print("Succesfully loaded weights.\n")
        print('_' * 60, "\n")


    train_dataloader, val_dataloader = setup_datasets_and_dataloaders(dataset_dir)
    
    if early_stopping:
        early_stopper = EarlyStopping(patience=5, path="checkpoints/best_model.pt")
    if experiment_name:
        experiment = setup_commet_loger(experiment_name)

    print("=================")
    print("STARTING TRAINING")
    print("=================")

    print(f"Using {device} device\n")
    print(f"The datset has {len(train_dataloader)} batches\n")
    for epoch in range(NUM_EPOCHS):
        kld_anneal_epochs = 75  # The number of epochs to ramp up the weight
        kld_max_weight = 0.002 

        if epoch < kld_anneal_epochs:
            # Linearly increase the weight from 0 to the max value
            kld_weight = kld_max_weight * (epoch / kld_anneal_epochs)
        else:
            kld_weight = kld_max_weight

        # Training phase
        model.train()
        train_loss, train_loss_reconstruction, train_loss_KL = 0, 0, 0

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
        for batch_idx, (sequences, lengths) in enumerate(train_dataloader):
            sequences = sequences.to(device)

            reconstructed_batch, mean, logvar = model(sequences, lengths)

            # Compute loss
            loss, loss_reconstruction, loss_KL = compute_loss(reconstructed_batch, sequences, mean, logvar, kld_weight=kld_weight)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_loss_reconstruction += loss_reconstruction.item()
            train_loss_KL += loss_KL.item()

            if verbose:
                if batch_idx % 10 == 0:
                    avg_train_loss = train_loss / (batch_idx + 1)
                    avg_train_loss_recon = train_loss_reconstruction / (batch_idx + 1)
                    avg_train_loss_KL = train_loss_KL / (batch_idx + 1)

                    print(f'\tBatch index: {batch_idx+1}/{len(train_dataloader)}')
                    print(f'\tCurrent training Loss: {avg_train_loss:.4f}')
                    print(f'\tCurrent training Reconstruction Loss: {avg_train_loss_recon:.4f}')
                    print(f'\tCurrent training KL Loss: {avg_train_loss_KL:.4f}')

                    if experiment_name:
                        experiment.log_metric("batch_train_loss", avg_train_loss, step=epoch * len(train_dataloader) + batch_idx)
                        experiment.log_metric("batch_train_loss_reconstruction", avg_train_loss_recon, step=epoch * len(train_dataloader) + batch_idx)
                        experiment.log_metric("batch_train_loss_KL", avg_train_loss_KL, step=epoch * len(train_dataloader) + batch_idx)
                    

        epoch_loss = train_loss / len(train_dataloader)
        epoch_reconstruction_loss = train_loss_reconstruction / len(train_dataloader)
        epoch_KL = train_loss_KL / len(train_dataloader)
        # Log epoch metrics
        if experiment:
            experiment.log_metric("epoch_train_loss", epoch_loss, step=epoch)
            experiment.log_metric("epoch_train_loss_reconstruction", epoch_reconstruction_loss, step=epoch)
            experiment.log_metric("epoch_train_loss_KL", epoch_KL, step=epoch)

        # Validation phase
        model.eval()
        val_loss, val_loss_reconstruction, val_loss_KL = 0, 0, 0
        print("Validating:")
        with torch.no_grad():
            for batch_idx, (sequences, lengths) in enumerate(val_dataloader):
                sequences = sequences.to(device)

                reconstructed_batch, mean, logvar = model(sequences, lengths)

                # Compute loss
                loss, loss_reconstruction, loss_KL = compute_loss(reconstructed_batch, sequences, mean, logvar, kld_weight=kld_weight)

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
            print('_' * 60, "\n")

        # Log validation metrics
        if experiment:
            experiment.log_metric("val_loss", val_epoch_loss, step=epoch)
            experiment.log_metric("epoch_val_loss_reconstruction", val_epoch_reconstruction_loss, step=epoch)
            experiment.log_metric("epoch_val_loss_KL", val_epoch_KL_loss, step=epoch)

        # Save model progress
        torch.save(model.state_dict(), f"./saved_models/progress/lofi-model_epoch{epoch+1}.pth")

        # Visualize some random samples reconstructed by the model
        if verbose:
            random_num = torch.randint(0, len(train_dataloader.dataset), (1,)).item()
            random_tensor, random_length = train_dataloader.dataset[random_num]
            print(f"\nVisualizing random sample {random_num} from the training dataset:")
            MidiDataset.visualize_midi(random_tensor)
            print("Reconstructed sample:")
            model.reconstruct(random_tensor, random_length)
            print('_' * 60, "\n")

        # Early stopping and saving the trained model
        if early_stopping:
            early_stopper(val_epoch_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

    torch.save(model.state_dict(), model_save_path)

    print("TRAINING FINISHED")
    print("===================")
    print(f"Best model was saved to: {model_save_path}")

    # End the Comet experiment
    if experiment:
        experiment.end()




