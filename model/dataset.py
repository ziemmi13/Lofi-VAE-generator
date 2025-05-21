from pretty_midi import PrettyMIDI
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import *
from torch.utils.data import random_split
import torch.nn.functional as F
from music21 import converter

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

        # print(f"{pianoroll_tensor.shape = }")

        max_len = MIDI_LEN
        item_len = pianoroll_tensor.shape[1] # shape is (pitch, time)
        if item_len < max_len:
            pad_len = max_len - item_len
            # print(f"{pad_len = }")
            pianoroll_tensor_ = F.pad(pianoroll_tensor, (0, pad_len), mode='constant', value=0)  # pad time axis
        else:
            pianoroll_tensor_ = pianoroll_tensor[:, :MIDI_LEN]
        
        
        # # Rescale for BCE
        # pianoroll_tensor_ = pianoroll_tensor_ / 127.0  # Rescale to [0, 1] for BCE (127 is max volume in MIDI)
        # pianoroll_tensor_ = torch.clamp(pianoroll_tensor_, 0, 1)


        # print(f"{pianoroll_tensor.shape = }")

        return pianoroll_tensor_

    def prepare_midi_file(self, file_path):
        """
        Prepares MIDI file to be procesed by the VAE model
        Args:
            filepath

        Returns:
            torch.Tensor: Tensor representation of the pianoroll
        
        """
        # Load MIDI file
        midi_file = PrettyMIDI(file_path)

        # Convert to pianoroll 
        pianoroll = midi_file.get_piano_roll(fs=24) 
        pianoroll = pianoroll[36:128, :]
        pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)#.T # Transpose to get (time, notes)
        #TODO
        # Check if changing pianoroll to Spectogram (also has note velocity) will benefit 'human feel'
        # Also check if dataset supports the idea of Spectogram

        # BINARIZE
        pianoroll_tensor = (pianoroll_tensor > 0).float()

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
    dataset = Dataset(dataset_dir)
    train_size = int(TRAIN_VALIDATION_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader

# Sanity check
# dataset = Dataset(dataset_dir=r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\Projekt zespolowy\dataset")
# print(dataset[0].shape)