import torch
from model.config import *
from dataset import setup_datasets_and_dataloaders

def train(model, dataset_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model.to(device)

    train_dataloader, val_dataloader = setup_datasets_and_dataloaders(dataset_dir)



