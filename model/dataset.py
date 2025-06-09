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
    def __init__(self, dataset_dir="dataset"):
        self.songs_dir = os.path.join(dataset_dir, "all_songs")
        self.df = pd.read_csv(os.path.join(dataset_dir, "midi_metadata.csv"))
        self.valid_instruments = [
            "drumms",
            "piano", # if piano not present check "organ"
            "bass",
            "guitar"
            # "others" - pick one present from "strings", "ensamble", "synth lead", "synth pad"
        ]

        self.midi_dict = RangeDict([
                (range(1, 9), "piano"),
                (range(9, 17), "chromatic percussion"),
                (range(17, 25), "organ"),
                (range(25, 33), "guitar"),
                (range(33, 41), "bass"),
                (range(41, 49), "strings"),
                (range(49, 57), "ensemble"),
                (range(57, 65), "brass"),
                (range(65, 73), "reed"),
                (range(73, 81), "pipe"),
                (range(81, 89), "synth lead"),
                (range(89, 97), "synth pad"),
                (range(97, 105), "synth effects"),
                (range(105, 113), "synth ethnic"),
                (range(113, 121), "synth percussive"),
                (range(121, 129), "sound effects"),
            ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index, verbose=False):
        # Load data from csv
        file_name = self.df.iloc[index, 0]
        file_path = os.path.join(self.songs_dir, file_name)
        bpm = int(round(self.df.iloc[index, 3]))

        pianoroll_tensor = self._prepare_pianoroll_tensor(file_path)
        seq_len = pianoroll_tensor.shape[2]
        if verbose:
            return pianoroll_tensor, seq_len, bpm, file_name
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
        midi_file = PrettyMIDI(file_path)

        # Convert to pianoroll for each instrument
        pianorolls = {instrument: None for instrument in self.valid_instruments}
        for instrument in midi_file.instruments:
            pianoroll = instrument.get_piano_roll(fs=FS) 
            print(f"{instrument = }")

            instrument_category = self.get_midi_instrument_name(instrument)
            # Fill the dict with PRESENT instruments
            if instrument_category in self.valid_instruments:
                pianoroll = pianoroll[MIN_MIDI_NOTE:MAX_MIDI_NOTE+1, :]
                pianorolls[instrument] = pianoroll

        # Add padding and fill in missing instruments 
        longest_instrument_track = int(midi_file.get_end_time()*FS)
        final_pianorolls = []
        for instrument in self.valid_instruments:
            pianoroll = pianorolls.get(instrument)
            # Fill in missing instruments with silence
            if pianoroll is None:
                silent_pianoroll = np.zeros((NUM_PITCHES, longest_instrument_track), dtype=np.float32)
                final_pianorolls.append(silent_pianoroll)
            # Add padding if necessary
            else:
                if pianoroll.shape[1] < longest_instrument_track:
                    pad_width = longest_instrument_track - pianoroll.shape[1]
                    padded_pianoroll = np.pad(pianoroll, ((0, 0), (0, pad_width)), mode='constant')
                    final_pianorolls.append(padded_pianoroll)

        # Stack pianorolls into a tensor
        pianoroll_stack = np.stack(final_pianorolls)

        # Reshape
        transposed_pianorolls = torch.tensor(pianoroll_stack, dtype=torch.float32)
        # Normalize velocities into <0,1> (max vel = 127) 
        transposed_pianorolls /= 127

        return transposed_pianorolls
    
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
    
    def get_midi_instrument_name(self, instrument):
        idx = instrument.program
        if instrument.is_drum:
            return "drumms" 
        else:
            return self.midi_dict[idx]
        
    
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

        # Max len in the batch
        num_instruments = tensors[0].shape[0]
        num_pitches = tensors[0].shape[1]
        max_len = max(lengths)

        # Create a batch of tensors of zeros for padding. All tensors will be padded to `max_len`.
        batch_size = len(tensors)
        padded_batch = torch.zeros(batch_size, num_instruments, num_pitches, max_len, dtype=torch.float32)
        
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
    dataset = MidiDataset(dataset_dir)
    train_size = int(TRAIN_VALIDATION_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MidiDataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MidiDataset.collate_fn)

    return train_dataloader, val_dataloader\
    
class RangeDict:
    def __init__(self, ranges):
        """
        Initialize with an iterable of (range, value) pairs.
        """
        self.ranges = list(ranges)

    def __getitem__(self, key):
        for rng, value in self.ranges:
            if key in rng:
                return value
        raise KeyError(f"{key} not found in any range.")

# Sanity check
dataset = MidiDataset()

for i in range(5):
    print(dataset[i][0].shape)
    print()
    print(50*"_")
    print()

