import torch
from train import train
from lofi_model import LofiModel
from dataset import MidiDataset
import pretty_midi
import numpy as np
from config import *
from utils import *

if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Dataset setup
    dataset_dir = r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\PROJEKT ZESPOLOWY\dataset\transformed_dataset"

    # Model setup
    model = LofiModel(device)
    model.to(device)
    print_model_details(model)

    # Train
    train(model, 
          dataset_dir, 
          experiment_name="more similar to mother project",
          verbose=True, 
          model_save_path = "./saved_models/more similar to mother project",
          weights_pth=None)
    
# 5 epok to 209 minut (1min utwór)
