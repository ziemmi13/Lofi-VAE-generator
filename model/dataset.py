from pretty_midi import PrettyMIDI
import torch
from torch.utils.data import Dataset
from glob import glob

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
        pianoroll = midi_file.get_piano_roll(fs=10) 
        pianoroll = pianoroll[21:109, :]
        pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)

        return pianoroll_tensor 


# dataset = Dataset(dataset_dir=r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\Projekt zespolowy\dataset")

# print(dataset[0])