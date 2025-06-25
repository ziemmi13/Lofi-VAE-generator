from pretty_midi import PrettyMIDI
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import *
from torch.utils.data import random_split
import torch.nn.functional as F
from music21 import converter, tempo
import os
import pandas as pd
import numpy as np

class MidiDataset(Dataset):
    def __init__(self, dataset_dir="dataset/transformed_dataset", verbose=False):
        self.songs_dir = os.path.join(dataset_dir, "all_songs")
        self.df = pd.read_csv(os.path.join(dataset_dir, "midi_metadata_clean.csv"))
        self.verbose = verbose
        self.count = 0
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # Load data from csv
        file_name = self.df.iloc[index, 0]
        file_path = os.path.join(self.songs_dir, file_name)
        bpm = int(round(self.df.iloc[index, 3]))

        pianoroll_tensor = self._prepare_pianoroll_tensor(file_path)
        seq_len = pianoroll_tensor.shape[2]
        if self.verbose:
            clean_file_name = os.path.splitext(file_name)[0]    
            return pianoroll_tensor, seq_len, bpm, clean_file_name
        return pianoroll_tensor, seq_len, bpm

    def _prepare_pianoroll_tensor(self, file_path):
        """
        Prepares MIDI file to be procesed by the VAE model
        Args:
            file_path (str): MIDI file path

        Returns:
            torch.Tensor: Tensor representation of the pianoroll (NUM_INSTRUMENTS, NUM_PITCHES, time).
        
        """
        # Load MIDI file
        try:
            midi_file = PrettyMIDI(file_path)
        except Exception as e:
            print(f"Error loading MIDI file {file_path}: {e}")
            return torch.zeros((5, NUM_PITCHES, 1), dtype=torch.float32)
        
        pianorolls = []
        # Convert to pianoroll for each instrument
        for instrument in midi_file.instruments:
            # pianoroll = pianoroll[MIN_MIDI_NOTE:MAX_MIDI_NOTE+1, :]

            # If the pianoroll is empty
            if len(instrument.notes) == 1:
                silent_pianoroll = np.zeros((NUM_PITCHES, 1), dtype=np.float32)
                pianorolls.append(silent_pianoroll)
            else:
                if instrument.is_drum:
                    pianoroll = self.drum_to_pianoroll(instrument)
                else:
                    pianoroll = instrument.get_piano_roll(fs=FS) 

                pianorolls.append(pianoroll)

    
        if len(midi_file.instruments) < 5:
            pianorolls = torch.zeros(NUM_INSTRUMENTS, NUM_PITCHES, 1)

        # Max instrument length
        max_instrument_len = max(pr.shape[1] for pr in pianorolls)
        actual_max_len = min(max_instrument_len, MAX_SEQ_LEN)

        # Pad all tracks to the same length (the actual_len of the song)
        padded_pianorolls = []
        for i, pr in enumerate(pianorolls):
            # Truncate if necessary
            pr_truncated = pr[:, :actual_max_len]
            
            # Pad if this specific track is shorter than the longest track in the song
            if pr_truncated.shape[1] < actual_max_len:
                pad_width = actual_max_len - pr_truncated.shape[1]
                padded_pr = np.pad(pr_truncated, ((0, 0), (0, pad_width)), mode='constant')
                padded_pianorolls.append(padded_pr)
            else:
                padded_pianorolls.append(pr_truncated)
        
        # Stack the consistently-sized pianorolls for this song
        pianoroll_stack = np.stack(padded_pianorolls)

        # Convert to tensor and normalize velocities to <0, 1>
        pianoroll_tensor = torch.tensor(pianoroll_stack, dtype=torch.float32)
        pianoroll_tensor /= 127.0 # Use float division

        return pianoroll_tensor
    
    def extract_chords(self, file_path):
        """
        Extracts chord data from MIDI file
        Args:
            file_path (str): MIDI file path
        Returns:
            list of lists: A list of chord data where: [root_note (str), quality (str), full_chord_name (str), offset (float)]
        """
        midi_file = converter.parse(file_path)

        chords = midi_file.chordify()

        chord_progression = []

        for c in chords.flat.getElementsByClass("Chord"):
            if not c.isRest:
                root_note = c.root().name
                full_chord_name = c.pitchedCommonName
                offset = c.offset  

                chord_progression.append([root_note, full_chord_name, offset])
        return chord_progression

    def drum_to_pianoroll(self, instrument):
        """
        Create a pianoroll for a drum track manually because for instrument.is_drum 
        the function: instrument.get_pianoroll() doesn't work.
        """
        end_time = max(note.end for note in instrument.notes)
        n_frames = int(end_time * FS) + 1
        pianoroll = np.zeros((NUM_PITCHES, n_frames)) 

        for note in instrument.notes:
            start = int(note.start * FS)
            end = int(note.end * FS)
            pitch = note.pitch
            velocity = note.velocity
            
            # Fill values in the piano roll
            pianoroll[pitch, start:end] = velocity

        return pianoroll

    
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

