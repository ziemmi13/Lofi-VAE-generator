from pretty_midi import PrettyMIDI
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import *
from torch.utils.data import random_split


class Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.filepaths = glob(f"{self.dataset_dir}/*.mid")
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of bounds")
        
        # Get midi file path
        midi_file_path = self.filepaths[index]
        # Prepare pianoroll tensor
        pianoroll_tensor = self.prepare_midi_file(midi_file_path)

        return pianoroll_tensor

    def prepare_midi_file(self, file_path):
        """
        Prepares MIDI file to be procesed by the VAE model
        Args:

        Returns:
        
        """
        # Load MIDI file
        midi_file = PrettyMIDI(file_path)

        # Convert to pianoroll 
        pianoroll = midi_file.get_piano_roll(fs=100) 
        pianoroll = pianoroll[21:109, :]
        pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)
        
        #TODO
        # Check if changing pianoroll to Spectogram (also has note velocity) will benefit 'human feel'
        # Also check if dataset supports the idea of Spectogram


        return pianoroll_tensor 

def setup_datasets_and_dataloaders(dataset_dir):
    dataset = Dataset(dataset_dir)
    train_size = int(TRAIN_VALIDATION_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    return train_dataloader, val_dataloader

# # Sanity check
dataset = Dataset(dataset_dir=r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\Projekt zespolowy\dataset")
print(dataset[0].shape)