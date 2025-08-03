from pretty_midi import PrettyMIDI
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from torch.utils.data import random_split
import os
import pandas as pd
from utils import drum_to_pianoroll
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence 
import matplotlib.pyplot as plt


class MidiDataset(Dataset):
    def __init__(self, dataset_dir=r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\PROJEKT ZESPOLOWY\dataset\maestro-piano-dataset", verbose=False):
        self.dataset_dir = dataset_dir
        all_midi_files = [f for f in os.listdir(dataset_dir)]
        self.df = pd.DataFrame(all_midi_files, columns=['file_name'])
        self.verbose = verbose
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # Load data from csv
        file_name = self.df.iloc[index, 0]
        file_path = os.path.join(self.dataset_dir, file_name)

        piano_tensor = self._prepare_pianoroll_tensor(file_path)
        seq_len = piano_tensor.shape[1]
        if self.verbose:
            clean_file_name = os.path.splitext(file_name)[0]    
            return piano_tensor, seq_len, clean_file_name
        
        return piano_tensor, seq_len

    def _prepare_pianoroll_tensor(self, file_path):
        """
        Prepares MIDI file to be procesed by the VAE model.
        Only drumms are considered.
        Args:
            file_path (str): MIDI file path

        Returns:
            torch.Tensor: Tensor representation of the pianoroll (NUM_PITCHES, time).
        
        """
        # Load MIDI file
        try:
            midi_file = PrettyMIDI(file_path)
        except Exception as e:
            print(f"Error loading MIDI file {file_path}: {e}")
            return torch.zeros((NUM_PITCHES, MAX_SEQ_LEN), dtype=torch.float32)
       
        # Convert midi file to pianoroll
        instrument = midi_file.instruments[0] 
        pianoroll = instrument.get_piano_roll(fs=FS)
       
        # Convert to tensor 
        pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)

        # Crop tensor to MIN_MIDI_NOTE and MAX_MIDI_NOTE
        pianoroll_tensor = pianoroll_tensor[MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1, :]

        pianoroll_len = pianoroll_tensor.shape[1]
        if pianoroll_len > MAX_SEQ_LEN:
            pianoroll_tensor = pianoroll_tensor[:, :MAX_SEQ_LEN]

        if pianoroll_len< MAX_SEQ_LEN:
            padding_needed = MAX_SEQ_LEN - pianoroll_len
            # The pad format is (pad_left, pad_right, pad_top, pad_bottom)
            # We only want to pad on the right of the time dimension (dim 1)
            pianoroll_tensor = F.pad(pianoroll_tensor, (0, padding_needed))

        # Normalize velocities to <0, 1>
        pianoroll_tensor /= 127.0 # Use float division

        return pianoroll_tensor
    

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[1], reverse=True)
        tensors, lengths= zip(*batch)

        # torch.stack is a more direct way to create the batch from a list of tensors
        padded_batch = torch.stack(tensors, dim=0)

        return (
            padded_batch, 
            torch.tensor(lengths, dtype=torch.long), 
        )
    
    @ staticmethod
    def visualize_midi(piano_roll):
        """
        Visualizes the MIDI piano roll.
        Args:
            piano_roll (torch.Tensor): Tensor representation of the pianoroll (NUM_PITCHES, time).
        """
        plt.figure(figsize=(12, 4))
        plt.imshow(piano_roll.numpy(), aspect='auto', origin='lower', cmap='hot')
        plt.xlabel('Time Steps')
        plt.ylabel('MIDI Notes')
        plt.title('Piano Roll Visualization')
        plt.colorbar(label='Velocity')
        plt.show()
        
def setup_datasets_and_dataloaders(dataset_dir):
    print("Setting up datasets and dataloaders...")
    dataset = MidiDataset(dataset_dir)
    train_size = int(TRAIN_VALIDATION_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MidiDataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MidiDataset.collate_fn)

    print("Finished setting up datasets and dataloaders.\n")
    return train_dataloader, val_dataloader

