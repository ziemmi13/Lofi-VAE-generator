import torch
from config import *
from dataset import setup_datasets_and_dataloaders
from loss import compute_loss

def train(model, dataset_dir, print_info=True, model_save_path = "./saved_models/lofi-model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_dataloader, val_dataloader = setup_datasets_and_dataloaders(dataset_dir)

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_loss_reconstruction = 0
        train_loss_KL = 0

        for batch_idx, data in enumerate(train_dataloader):
            data = data.to(device)
       
            optimizer.zero_grad()

            distribution, z, reconstructed_data = model(data)

            # print(f"{z.shape = }")
            # print(f"{reconstructed_data.shape = }")
            # print(f"{data.shape = }")

            # Compute loss
            loss, loss_reconstruction, loss_KL = compute_loss(data, reconstructed_data,  distribution, z, loss_type = "BCE")
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Compute loss
            train_loss += loss.item()
            train_loss_reconstruction += loss_reconstruction.item()
            train_loss_KL += loss_KL.item()

            # Update network
            optimizer.step()

            # Print info
            if print_info:
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                    # experiment.log_metric("train_loss", loss.item(), step=epoch * len(train_loader) + batch_idx)
                    # experiment.log_metric("train_loss_reconstruction", loss_reconstruction.item(), step=epoch * len(train_loader) + batch_idx)
                    # experiment.log_metric("train_loss_KL", loss_KL.item(), step=epoch * len(train_loader) + batch_idx)
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
            for batch, data in enumerate(val_dataloader):
                data = data.to(device)

                distribution, z, reconstructed_data = model(data)


                # Compute loss
                loss, loss_reconstruction, loss_KL = compute_loss(data, reconstructed_data,  distribution, z, loss_type = "BCE")

                val_loss += loss.item()
                val_loss_reconstruction += loss_reconstruction.item()
                val_loss_KL += loss_KL.item()

        val_epoch_loss = val_loss / len(val_dataloader) 
        val_epoch_reconstruction_loss = val_loss_reconstruction / len(val_dataloader)
        val_epoch_KL_loss = val_loss_KL / len(val_dataloader)
        
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {val_epoch_loss:.4f}')
        print(f'Reconstruction Loss: {val_epoch_reconstruction_loss:.4f}')
        print(f'KL Loss: {val_epoch_KL_loss:.4f}')
        print('-' * 60)

        # # Log validation metrics
        # experiment.log_metric("val_loss", val_epoch_loss, step=epoch)
        # experiment.log_metric("epoch_val_loss_reconstruction", val_epoch_reconstruction_loss, step=epoch)
        # experiment.log_metric("epoch_val_loss_KL", val_epoch_KL, step=epoch)

    # Saving the trained model

    print("Finished training!")
    print(f"Saving the model to path: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

    print("Model saved!")




