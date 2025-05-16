import torch
from train import train
from lofi_model import LofiModel

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Dataset setup
    dataset_dir = r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\Projekt zespolowy\dataset"

    # Model setup
    model = LofiModel(device)

    # Train
    train(model, dataset_dir, print_info=True)
