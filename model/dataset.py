from pretty_midi import PrettyMIDI
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from config import *
from torch.utils.data import random_split
import torch.nn.functional as F
from music21 import converter, tempo
import os

class MidiDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.filepaths = glob(f"{self.dataset_dir}/*.mid")

        # A cache to store pre-processed (tensor, length) tuples.
        self.data_cache = [] 
        self._preprocess_data() 
    
    def _preprocess_data(self):
        """
        Pre-processes all MIDI files found and stores their tensor representations
        and original lengths in memory (self.data_cache).
        This method filters out invalid or too-long/too-short sequences.
        It also calculates the total number of 1s and 0s for loss weighting.
        """
        print(f"Starting MIDI dataset preprocessing from: {self.dataset_dir}")
        print(f"Found {len(self.filepaths)} MIDI files.")

        num_skipped_length = 0 # Counter for sequences skipped due to length filters
        num_skipped_error = 0 

        for i, file_path in enumerate(self.filepaths):
            # Print progress periodically to keep track of long preprocessing jobs.
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(self.filepaths)} files.")
            pianoroll_tensor = self._prepare_pianoroll_tensor(file_path)
            
            if pianoroll_tensor is not None:
                seq_len = pianoroll_tensor.shape[1]
            
                # Apply length filters.
                if MIN_SEQ_LEN_FILTER <= seq_len <= MAX_SEQ_LEN_FILTER:
                    # PrettyMIDI's get_piano_roll returns (pitch, time).
                    # For LSTMs with `batch_first=True`, we need (time, pitch).
                    # So, we transpose the tensor before caching it.
                    transposed_tensor = pianoroll_tensor.T
                    self.data_cache.append((transposed_tensor, seq_len, file_path))
                else:
                        num_skipped_length += 1
                        print(f"  Skipping '{os.path.basename(file_path)}': Length {seq_len} not in [{MIN_SEQ_LEN_FILTER}, {MAX_SEQ_LEN_FILTER}].")
            else:
                num_skipped_error += 1
        
        print(f"Finished preprocessing. Total valid sequences: {len(self.data_cache)}")
        print(f"  Skipped {num_skipped_length} files due to length filters.")
        print(f"  Skipped {num_skipped_error} files due to processing errors or being empty.")
        print(50*"_"+"\n")
    
    def __len__(self):
        return len(self.data_cache)
    
    def __getitem__(self, index):
        return self.data_cache[index]

    def _prepare_pianoroll_tensor(self, file_path):
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
        pianoroll = pianoroll[MIN_MIDI_NOTE:MAX_MIDI_NOTE+1, :]
        pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)

        # BINARIZE
        pianoroll_tensor = (pianoroll_tensor > 0).float()

        return pianoroll_tensor

    # def pad_midi_tensor(self, pianoroll_tensor):
    #     num_pitches, item_len = pianoroll_tensor.shape
    #     if item_len < MIDI_LEN:
    #         pad_len = MIDI_LEN - item_len
    #         # Pad the time axis
    #         pianoroll_tensor = F.pad(pianoroll_tensor, (0, pad_len, 0, 0), mode='constant', value=0)
    #     else:
    #         pianoroll_tensor = pianoroll_tensor[:, :MIDI_LEN]
        
    #     return pianoroll_tensor
    
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
    
    def get_bpm(self, file_path):
        score = converter.parse(file_path)
        for el in score.recurse():
            if isinstance(el, tempo.MetronomeMark):
                return el.number
            else:
                return 90
                # TODO
                # Convert files to wav and extract bpm using librosa
                # Save to csv
                # Read from csv
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for PyTorch DataLoader.
        This is CRUCIAL for handling variable-length sequences in a batch.
        It pads all sequences in a batch to the length of the longest sequence
        in that specific batch. It also sorts the batch by length, which is
        often recommended for efficiency with `pack_padded_sequence`.
        """
        # Sort the batch by sequence length in descending order.
        # This is a common optimization for `pack_padded_sequence` in LSTMs.
        # `x[1]` refers to the original length stored in the tuple `(tensor, length)`.
        batch.sort(key=lambda x: x[1], reverse=True)

        sequences = [item[0] for item in batch] # Extract the tensors
        lengths = [item[1] for item in batch]   # Extract the original lengths

        # Determine the maximum sequence length in the current batch.
        max_len = max(lengths)
        # Get the number of features (notes), which should be INPUT_DIM (88).
        num_features = sequences[0].shape[1] 

        # Create a tensor of zeros for padding. All sequences will be padded to `max_len`.
        padded_sequences = torch.zeros(len(batch), max_len, num_features, dtype=torch.float32)
        
        # Fill the padded tensor with the actual sequence data.
        for i, seq in enumerate(sequences):
            # Place the sequence at the beginning of its row, leaving the rest as padding (zeros).
            padded_sequences[i, :len(seq), :] = seq

        # Return the padded sequences and their original lengths.
        # The lengths tensor is essential for `pack_padded_sequence`.
        return padded_sequences, torch.tensor(lengths, dtype=torch.long)



def setup_datasets_and_dataloaders(dataset_dir):
    dataset = MidiDataset(dataset_dir)
    train_size = int(TRAIN_VALIDATION_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MidiDataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=MidiDataset.collate_fn)

    return train_dataloader, val_dataloader

# Sanity check
# dataset = MidiDataset(dataset_dir=r"C:\Users\Hyperbook\Desktop\STUDIA\SEM III\Projekt zespolowy\dataset")
# print(dataset[1])