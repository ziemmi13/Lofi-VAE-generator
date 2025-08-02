from pretty_midi import PrettyMIDI
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from torch.utils.data import random_split
import os
import pandas as pd
from utils import drum_to_pianoroll

class MidiDataset(Dataset):
    def __init__(self, dataset_dir=r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\PROJEKT ZESPOLOWY\dataset\golden_dataset", verbose=False):
        self.songs_dir = os.path.join(dataset_dir, "all_songs")
        self.df = pd.read_csv(os.path.join(dataset_dir, "midi_metadata.csv"))
        self.verbose = verbose
        self.count = 0
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # Load data from csv
        file_name = self.df.iloc[index, 0]
        file_path = os.path.join(self.songs_dir, file_name)
        bpm = int(round(self.df.iloc[index, 3]))

        drumms_tensor = self._prepare_pianoroll_tensor(file_path)
        seq_len = drumms_tensor.shape[1]
        if self.verbose:
            clean_file_name = os.path.splitext(file_name)[0]    
            return drumms_tensor, seq_len, bpm, clean_file_name
        return drumms_tensor, seq_len, bpm

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
        
        # Convert to pianoroll for each instrument
        for instrument in midi_file.instruments:
            if instrument.is_drum:
                pianoroll = drum_to_pianoroll(instrument)

        # Convert to tensor and normalize velocities to <0, 1>
        pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)
        pianoroll_tensor /= 127.0 # Use float division

        return pianoroll_tensor
    
    @staticmethod
    def collate_fn(batch):
        """
         It pads all sequences in a batch to the length of the longest sequence.
        Args:
            batch (list): A list of tuples, where each tuple is 
                          (pianoroll_tensor, seq_len, bpm).
                          pianoroll_tensor shape: (num_instruments, num_pitches, time)
        Returns:
            tuple: A tuple containing:
                - padded_sequences (torch.Tensor): Padded tensors of shape 
                  (batch_size, num_instruments, num_pitches, max_len).
                - lengths (torch.Tensor): Original sequence lengths of shape (batch_size,).
                - bpms (torch.Tensor): BPM values for each item in the batch (batch_size,).
        """
        # Sort the batch by sequence length in descending order.
        # This is a common optimization for `pack_padded_sequence` in LSTMs.
        # `x[1]` refers to the original length stored in the tuple `(tensor, length)`.
        batch.sort(key=lambda x: x[1], reverse=True)

        # Extract tensors, lenghts, bpms
        tensors, lengths, bpms = zip(*batch)

        # Create a batch of tensors of zeros for padding. All tensors will be padded to `max_len`.
        batch_size = len(tensors)
        padded_batch = torch.zeros(batch_size, NUM_INSTRUMENTS, NUM_PITCHES, MAX_SEQ_LEN, dtype=torch.float32)
        
        # Fill the padded tensor with the actual sequence data.
        for i, tensor in enumerate(tensors):
            # Get the original length of the current sequence
            end = lengths[i]
            # Use slicing to copy the 3D tensor into its place in the 4D batch tensor
            padded_batch[i, :, :, :end] = tensor

        return (
            padded_batch, 
            torch.tensor(lengths, dtype=torch.long), 
            torch.tensor(bpms, dtype=torch.long)
        )

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

