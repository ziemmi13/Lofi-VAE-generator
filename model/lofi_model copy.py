import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pretty_midi
from config import *
from dataset import MidiDataset
import os

class LofiModel(nn.Module):
    def __init__(self, device, num_instruments=NUM_INSTRUMENTS, num_pitches=NUM_PITCHES):
        super(LofiModel, self).__init__()
        self.device = device
        self.num_instruments = num_instruments
        self.num_pitches = num_pitches
        
        # Calculate the flattened input dimension
        self.input_dim = self.num_instruments * self.num_pitches

        # The Encoder and Decoder are instantiated with the new flattened input_dim
        self.encoder = Encoder(input_dim=self.input_dim, hidden_dim=LATENT_DIM, latent_dim=LATENT_DIM, num_layers=LSTM_LAYERS)
        self.decoder = Decoder(lstm_input_dim=self.input_dim,
                               lstm_hidden_dim=LATENT_DIM,
                               z_latent_dim=LATENT_DIM,
                               output_dim=self.input_dim,
                               num_layers=LSTM_LAYERS)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        z = mu + eps * std
        return z
        
    def forward(self, x, lengths):
        """
        Performs a forward pass of VAE.
        
        Args:
            x (torch.Tensor): Input pianorolls (batch_size, num_instruments, num_pitches, max_len).
            lengths (torch.Tensor): Original sequence lengths in time steps.

        Returns:
            reconstructed_x (torch.Tensor): Reconstructed pianorolls (batch_size, num_instruments, num_pitches, max_len).
            mu (torch.Tensor): Latent mean (batch_size, latent_dim).
            logvar (torch.Tensor): Latent log variance (batch_size, latent_dim).
        """
        batch_size, _, _, max_len = x.shape

        # Reshape from (batch_size, num_instruments, num_pitches, max_len) to (batch_size, max_len, num_instruments*num_pitches, ) for LSTM
        x_flat = x.permute(0, 3, 1, 2).contiguous() # -> (B, T, I, P)

        x_flat = x_flat.view(batch_size, max_len, -1) # -> (B, T, I*P)
        
        # Encoder and Reparameterization
        mu, logvar = self.encoder(x_flat, lengths)
        z = self.reparameterize(mu, logvar)
        
        # Decoder (teacher forcing with the flattened ground truth)
        reconstructed_x_flat = self.decoder(x_flat, z, lengths) # -> (B, T, I*P)

        #Reshape back to (batch_size, num_instruments, num_pitches, max_len)
        reconstructed_x = reconstructed_x_flat.view(batch_size, max_len, self.num_instruments, self.num_pitches) # -> (B, T, I, P)
        reconstructed_x = reconstructed_x.permute(0, 2, 3, 1).contiguous() # -> (B, I, P, T)

        return reconstructed_x, mu, logvar

    def reconstruct(self, x, lengths, max_length=None):
        """
        Reconstructs an input sequence without teacher forcing (used for generation/evaluation).

        Args:
            x (torch.Tensor): Input pianorolls of shape (B, I, P, T).
            lengths (torch.Tensor): Sequence lengths (B).
            max_length (int, optional): Max length to decode. If None, use max from lengths.

        Returns:
            reconstructed_x (torch.Tensor): Reconstructed pianoroll (B, I, P, T).
        """
        batch_size, _, _, seq_len = x.shape
        max_length = max_length if max_length is not None else seq_len

        # Reshape input: (B, I, P, T) → (B, T, I*P)
        x_flat = x.permute(0, 3, 1, 2).contiguous().view(batch_size, seq_len, -1)

        # Encode
        mu, logvar = self.encoder(x_flat, lengths)
        z = self.reparameterize(mu, logvar)

        # Decode step-by-step using LSTMCell-style generation
        generated_flat = self.decoder.generate_from_latent(z, max_length=max_length)  # (B, max_length, I*P)

        # Reshape: (B, max_len, I*P) → (B, I, P, T)
        reconstructed = generated_flat.view(batch_size, max_length, self.num_instruments, self.num_pitches)
        reconstructed = reconstructed.permute(0, 2, 3, 1).contiguous()  # (B, I, P, T)

        return reconstructed



    
    def tensor_to_midi(self, tensor, save_to_midi, save_path):
            midi_file = pretty_midi.PrettyMIDI()
            
            # Instrument mapping for midi 
            instrument_map = [
                {'program': 0,  'is_drum': True,  'name': 'Drums'},      # 0: Standard Drum Kit
                {'program': 0,  'is_drum': False, 'name': 'Piano'},      # 0: Acoustic Grand Piano
                {'program': 33, 'is_drum': False, 'name': 'Bass'},       # 33: Electric Bass (finger)
                {'program': 25, 'is_drum': False, 'name': 'Guitar'},     # 25: Acoustic Guitar (steel)
                {'program': 48, 'is_drum': False, 'name': 'Others'}      # 48: String Ensemble 1
            ]

            print(f"{tensor.shape=}, {tensor.dtype=}, {tensor.device=}")

            # Iterate through each instrument's pianoroll in the givem tensor
            for i in range(tensor.shape[0]):
                instrument_pianoroll = tensor[i, :, :]
                instrument_info = instrument_map[i]
                
                instrument = pretty_midi.Instrument(
                    program=instrument_info['program'], 
                    is_drum=instrument_info['is_drum'],
                    name=instrument_info['name']
                )
                
                # Convert pioanoroll to MIDI 
                pr_T = instrument_pianoroll.cpu().numpy().T 

                for pitch_idx in range(pr_T.shape[1]):
                    note_on_time = None
                    peak_velocity_normalized = 0.0
                    for t in range(pr_T.shape[0]):
                        current_velocity_normalized = pr_T[t, pitch_idx]
                        # Use a threshold to decide if a note is "on"
                        is_note_on = current_velocity_normalized > 0.1

                        # --- Note On Event ---
                        if is_note_on and note_on_time is None:
                            note_on_time = t / FS # Convert time step to seconds
                            peak_velocity_normalized = current_velocity_normalized
                        
                        # --- Note Continues ---
                        elif is_note_on and note_on_time is not None:
                            # Update the peak velocity if the current one is higher
                            if current_velocity_normalized > peak_velocity_normalized:
                                peak_velocity_normalized = current_velocity_normalized

                        # --- Note Off Event ---
                        elif not is_note_on and note_on_time is not None:
                            note_off_time = t / FS
                            velocity_midi = int(peak_velocity_normalized * 127)
                            
                            # Add note only if velocity is significant
                            if velocity_midi > 0:
                                note = pretty_midi.Note(
                                    velocity=min(127, velocity_midi),
                                    pitch=pitch_idx + MIN_MIDI_NOTE,
                                    start=note_on_time,
                                    end=note_off_time
                                )
                                instrument.notes.append(note)
                            
                            # Reset for the next note
                            note_on_time = None
                            peak_velocity_normalized = 0.0
                    
                    # After the loop, close any note that's still on at the very end
                    if note_on_time is not None:
                        note_off_time = pr_T.shape[0] / FS
                        velocity_midi = int(peak_velocity_normalized * 127)
                        if velocity_midi > 0:
                            note = pretty_midi.Note(
                                velocity=min(127, velocity_midi),
                                pitch=pitch_idx + MIN_MIDI_NOTE,
                                start=note_on_time,
                                end=note_off_time
                            )
                            instrument.notes.append(note)

                # Add the completed instrument to the MIDI file
                midi_file.instruments.append(instrument)

            # Save MIDI file
            if save_to_midi:
                midi_file.write(save_path)
                print(f"MIDI saved to {save_path}")
            
            return midi_file


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout=0.3):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)

        self.hidden_to_mu = nn.Linear(hidden_dim * num_layers, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim * num_layers, latent_dim)

    def forward(self, x, lengths):
        # Pack padded batch of sequences for the LSTM
        # Ensure lengths are on CPU and are integers/long for pack_padded_sequence
        packed_input = pack_padded_sequence(x, lengths.cpu().long(), batch_first=True, enforce_sorted=True) # NOTE: Enforce sorted for efficiency

        # Pass the packed input through the LSTM.
        # hidden and cell are the final states: (num_layers * num_directions, batch, hidden_size)
        _, (hidden, cell) = self.lstm(packed_input)
        # Flatten hidden layers: (num_layers, batch, hidden_dim) -> (batch, num_layers * hidden_dim)
        # LSTM's hidden is (D*num_layers, N, H_out) where D=1 for non-bidirectional
        # Permute to (N, D*num_layers, H_out) then view as (N, D*num_layers*H_out)
        hidden = hidden.permute(1, 0, 2).contiguous().view(x.size(0), -1)
        # mu and logvar
        mu = self.hidden_to_mu(hidden)
        logvar = self.hidden_to_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, z_latent_dim, output_dim, num_layers, dropout=0.3):
        super(Decoder, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.lstm_input_dim = lstm_input_dim    
        self.latent_to_hidden = nn.Linear(z_latent_dim, num_layers * lstm_hidden_dim)
        self.latent_to_cell = nn.Linear(z_latent_dim, num_layers * lstm_hidden_dim)
        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                              hidden_size=lstm_hidden_dim,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        
        # Output layer: maps LSTM hidden state to output feature dimension
        self.fc_out = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x_teacher_force, z, lengths, input_dropout_p=0.2):
        batch_size = z.size(0)
        max_len = x_teacher_force.shape[1]
        
        # Project latent vector z to initial hidden and cell states
        h_0 = self.latent_to_hidden(z).view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = self.latent_to_cell(z).view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()
        
        # --- NEW: APPLY INPUT DROPOUT ---
        # Create a dropout mask
        # We want to drop out entire time steps, not individual notes
        dropout_mask = (torch.rand(batch_size, max_len, 1, device=x_teacher_force.device) > input_dropout_p).float()
        
        # Apply the mask to the teacher-forcing input
        # This will set entire time-step vectors to zero
        corrupted_input = x_teacher_force * dropout_mask
        
        # Pack the *corrupted* input sequence
        packed_input = pack_padded_sequence(corrupted_input, lengths.cpu().long(), batch_first=True, enforce_sorted=False)

        # LSTM forward pass
        packed_output, _ = self.lstm(packed_input, (h_0, c_0))
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)
        
        # Pass LSTM outputs through the final fully connected layer
        reconstructed_x = self.fc_out(output)
        return reconstructed_x
    
    def generate_from_latent(self, z, max_length):
        batch_size = z.size(0)
        
        # Zakładamy: self.latent_to_hidden i self.latent_to_cell istnieją
        h_t = self.latent_to_hidden(z).view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()
        c_t = self.latent_to_cell(z).view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()

        # Start token - np. wektor zerowy albo wyuczony token
        input_t = torch.zeros(batch_size, 1, INPUT_DIM, device=z.device)  # (B, 1, input_dim)

        outputs = []

        for _ in range(max_length):
            output_t, (h_t, c_t) = self.lstm(input_t, (h_t, c_t))
            out = self.fc_out(output_t)  # (B, 1, output_dim)

            # Można użyć sampling lub argmax
            sampled_output = torch.sigmoid(out)  # jeśli BCE na wyjściu
            sampled_output = (sampled_output > 0.5).float()

            outputs.append(sampled_output.squeeze(1))  # usuń dim seq_len=1

            # Użyj poprzedniego wyjścia jako nowego wejścia
            input_t = sampled_output.detach()  # lub sampled_output

        outputs = torch.stack(outputs, dim=1)  # (B, max_length, output_dim)

        return outputs

    