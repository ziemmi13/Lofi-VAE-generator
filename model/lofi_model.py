import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pretty_midi
from config import *
from dataset import MidiDataset
import os
from utils import tensor_to_midi

class LofiModel(nn.Module):
    def __init__(self, device, num_instruments=NUM_INSTRUMENTS, num_pitches=NUM_PITCHES):
        super(LofiModel, self).__init__()
        self.device = device
        self.num_instruments = num_instruments
        self.num_pitches = num_pitches
        
        # Dimensions
        self.input_dim = self.num_instruments * self.num_pitches
        self.hidden_dim = LATENT_DIM # Using LATENT_DIM as hidden_dim for simplicity

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size=self.input_dim,
                                     hidden_size=self.hidden_dim,
                                     batch_first=True)
        self.fc_mu = nn.Linear(self.hidden_dim, LATENT_DIM)
        self.fc_logvar = nn.Linear(self.hidden_dim, LATENT_DIM)

        # Decoder
        self.decoder_lstm = nn.LSTM(input_size=LATENT_DIM,
                                     hidden_size=self.hidden_dim,
                                     batch_first=True)
        self.fc_output = nn.Linear(self.hidden_dim, self.input_dim)

        self.decoder_input_dim = self.input_dim + LATENT_DIM  # concatenated input + context z
        self.decoder_cell = nn.LSTMCell(self.decoder_input_dim, LATENT_DIM)
    
    def encode(self, x, lengths):
        batch_size, _, _, max_len = x.shape
        # Reshape from (batch_size, num_instruments, num_pitches, max_len) to (batch_size, max_len, num_instruments*num_pitches)
        x_flat = x.permute(0, 3, 1, 2) # -> (B, T, I, P)
        x_flat = x_flat.view(batch_size, max_len, -1) # -> (B, T, I*P)
        
        # Pack padded sequences for LSTM
        packed_input = pack_padded_sequence(x_flat, lengths.cpu().long(), batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        # We only need the final hidden state
        _, (hidden, _) = self.encoder_lstm(packed_input)
        hidden = hidden.squeeze(0)  
        
        # Compute mu and logvar
        mu = self.fc_mu(hidden)  
        logvar = self.fc_logvar(hidden)
        return mu, logvar        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        z = mu + eps * std
        return z
    
    def decode(self, z, input_seq, teacher_forcing_ratio=0.5):
        # input_seq: (B, I, P, T) — ground truth used for teacher forcing
        # z: (B, latent_dim)

        batch_size, num_instr, num_pitch, seq_len = input_seq.shape
        device = z.device

        # Reshape input_seq to (B, T, I*P)
        input_seq = input_seq.permute(0, 3, 1, 2).contiguous().view(batch_size, seq_len, -1)

        # Initial input token — zero or learned start token
        input_t = torch.zeros(batch_size, self.input_dim, device=device)

        # Initial LSTM states
        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=device)

        outputs = []

        for t in range(seq_len):
            # Concatenate input token with context vector z
            decoder_input = torch.cat([input_t, z], dim=1)  # Shape: (B, input_dim + latent_dim)

            # Pass through LSTMCell
            h_t, c_t = self.decoder_cell(decoder_input, (h_t, c_t))

            # Output projection
            out_t = self.fc_output(h_t)  # Shape: (B, I*P)
            outputs.append(out_t.unsqueeze(1))  # (B, 1, I*P)

            # Decide whether to use teacher forcing
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio

            if use_teacher:
                input_t = input_seq[:, t, :]  # Use ground truth
            else:
                input_t = out_t.detach()  # Use model output (detach to avoid backprop through sampling)

        # Concatenate outputs: (B, T, I*P)
        outputs = torch.cat(outputs, dim=1)

        # Reshape to (B, I, P, T)
        outputs = outputs.view(batch_size, seq_len, num_instr, num_pitch)
        outputs = outputs.permute(0, 2, 3, 1)  # (B, I, P, T)

        return outputs
        
    def forward(self, x, lengths):
        # Encode the input sequence to get the parameters of the latent distribution
        mu, logvar = self.encode(x, lengths)
        
        # Sample from the latent distribution using the reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode the latent vector to reconstruct the original sequence
        reconstructed_x = self.decode(z, x)
        
        return reconstructed_x, mu, logvar

    def reconstruct(self, x, lengths, bpm=90, save_path="reconstructed.mid"):
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
             # Prepare tensor
            tensor_len = x.shape[2]
            padded_tensor, tensor_len, _ = MidiDataset.collate_fn([(x, tensor_len, bpm)])
            
            padded_tensor = padded_tensor.to(self.device)
            tensor_len = tensor_len.to(self.device)

            # Reconstruct sample
            reconstructed_sample, _, _ = self(padded_tensor, tensor_len)
            reconstructed_sample = reconstructed_sample.squeeze()

            # Prepare for MIDI conversion
            reconstructed_sample = torch.clamp(reconstructed_sample, 0.0, 1.0) # Normalize to [0, 1]
            reconstructed_sample[reconstructed_sample < 0.05] = 0.0 # Threshold to remove noise

            # Convert tensor to MIDI file
            midi_file = tensor_to_midi(reconstructed_sample, save_to_midi=True, save_path=save_path)
            return reconstructed_sample, midi_file

        return reconstructed



