from pretty_midi import PrettyMIDI
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import *
from torch.utils.data import random_split
import torch.nn.functional as F
from music21 import converter

class MidiDataset(Dataset):
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

        # Add padding if necesarry
        padded_pianoroll_tensor = self.pad_midi_tensor(pianoroll_tensor)
        
        # Rescale for BCE
        # padded_pianoroll_tensor = padded_pianoroll_tensor / 127.0  # Rescale to [0, 1] for BCE (127 is max volume in MIDI)
        # padded_pianoroll_tensor = torch.clamp(padded_pianoroll_tensor, 0, 1) # Ensure the range

        # Change from (num_pitches, MIDI_LEN) to (MIDI_LEN, num_pitches) for LSTM
        permuted_tensor = padded_pianoroll_tensor.permute(1, 0)

        return permuted_tensor

    def prepare_midi_file(self, file_path):
        """
        Prepares MIDI file to be procesed by the VAE model
        Args:
            file_path (str): MIDI file path

        Returns:
            torch.Tensor: Tensor representation of the pianoroll
        
        """
        # Load MIDI file
        midi_file = PrettyMIDI(file_path)

        # Convert to pianoroll 
        pianoroll = midi_file.get_piano_roll(fs=FS) 
        pianoroll = pianoroll[36:85, :] # Crop the piano roll (C2 - C6)
        pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)
        #TODO
        # Check if changing pianoroll to Spectogram (also has note velocity) will benefit 'human feel'
        # Also check if dataset supports the idea of Spectogram

        # BINARIZE
        pianoroll_tensor = (pianoroll_tensor > 0).float()

        return pianoroll_tensor

    def pad_midi_tensor(self, pianoroll_tensor):
        num_pitches, item_len = pianoroll_tensor.shape
        if item_len < MIDI_LEN:
            pad_len = MIDI_LEN - item_len
            # Pad the time axis
            pianoroll_tensor = F.pad(pianoroll_tensor, (0, pad_len, 0, 0), mode='constant', value=0)  # pad time axis
        else:
            pianoroll_tensor = pianoroll_tensor[:, :MIDI_LEN]
        
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
                quality = c.quality
                full_chord_name = c.pitchedCommonName
                offset = c.offset  

                chord_progression.append([root_note, quality, full_chord_name, offset])
        return chord_progression


def setup_datasets_and_dataloaders(dataset_dir):
    dataset = MidiDataset(dataset_dir)
    train_size = int(TRAIN_VALIDATION_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader

# Sanity check
# dataset = MidiDataset(dataset_dir=r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\Projekt zespolowy\dataset")
# print(dataset[0].shape)