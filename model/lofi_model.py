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

    def reconstruct(self, midi_tensor, bpm, save_to_midi=True, save_path="reconstructed/"):
        self.eval()
        with torch.no_grad():
            # Prepare tensor
            tensor_len = midi_tensor.shape[2]
            padded_tensor, tensor_len, _ = MidiDataset.collate_fn([(midi_tensor, tensor_len, bpm)])
            
            padded_tensor = padded_tensor.to(self.device)
            tensor_len = tensor_len.to(self.device)

            # Reconstruct sample
            reconstructed_sample, _, _ = self(padded_tensor, tensor_len)
            reconstructed_sample = reconstructed_sample.squeeze()

            # Prepare for MIDI conversion
            reconstructed_sample = torch.clamp(reconstructed_sample, 0.0, 1.0) # Normalize to [0, 1]
            reconstructed_sample[reconstructed_sample < 0.05] = 0.0 # Threshold to remove noise

            # Convert tensor to MIDI file
            midi_file = self.tensor_to_midi(reconstructed_sample, save_to_midi, save_path)
            return reconstructed_sample, midi_file if save_to_midi else None

    def generate(self, num_samples, max_len, z_sample=None, temperature=1.0, threshold=0.01, save_to_midi=True, save_path="generated_sample.mid"):
        self.eval()
        with torch.no_grad():
            # 1. --- FIX: Ensure z_sample has a batch dimension ---
            if z_sample is None:
                # Shape is now (num_samples, LATENT_DIM)
                z_sample = torch.randn(num_samples, LATENT_DIM).to(self.device)
            else:
                z_sample = z_sample.to(self.device)
                if z_sample.dim() == 1: # Add batch dim if a single z is provided
                    z_sample = z_sample.unsqueeze(0)
            
            # 2. Generate from latent vector
            # The decoder now handles the batch dimension correctly
            generated_flat = self.decoder.generate_from_latent(z_sample, max_len, self.device, temperature)

            # 3. Reshape back to the pianoroll format (B, I, P, T)
            generated_pianoroll = generated_flat.view(num_samples, max_len, self.num_instruments, self.num_pitches)
            generated_pianoroll = generated_pianoroll.permute(0, 2, 3, 1).contiguous()

            # Process and save each sample in the batch
            midi_files = []
            for i in range(num_samples):
                single_pianoroll = generated_pianoroll[i] # Get the i-th sample
                
                # Threshold to remove noise
                single_pianoroll[single_pianoroll < threshold] = 0.0

                if save_to_midi:
                    # Adjust save path for multiple samples
                    current_save_path = save_path
                    if num_samples > 1:
                        path_parts = os.path.splitext(save_path)
                        current_save_path = f"{path_parts[0]}_{i}{path_parts[1]}"

                    midi_file = self.tensor_to_midi(single_pianoroll, save_to_midi=True, save_path=current_save_path)
                    midi_files.append(midi_file)
            
            # Return the generated tensor and the list of MIDI files
            return generated_pianoroll, midi_files if save_to_midi else None

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
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout=0.1):
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
    def __init__(self, lstm_input_dim, lstm_hidden_dim, z_latent_dim, output_dim, num_layers, dropout=0.1):
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

    def forward(self, x_teacher_force, z, lengths):
        batch_size = z.size(0)
        max_len = x_teacher_force.shape[1]
        
        # Project latent vector z to initial hidden and cell states
        # Output shape: (batch_size, num_layers * lstm_hidden_dim)
        initial_hidden_flat = self.latent_to_hidden(z)
        initial_cell_flat = self.latent_to_cell(z)
        
        # Reshape to (num_layers, batch_size, lstm_hidden_dim) for LSTM
        h_0 = initial_hidden_flat.view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = initial_cell_flat.view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()
        
        # Prepare input sequence for LSTM (using teacher forcing with x_teacher_force)
        # x_teacher_force has shape (batch_size, max_seq_len, rnn_input_dim)
        # Ensure lengths are on CPU and are integers/long for pack_padded_sequence
        packed_input = pack_padded_sequence(x_teacher_force, lengths.cpu().long(), batch_first=True, enforce_sorted=True) # NOTE: Enforce sorted for efficiency

        # LSTM forward pass
        packed_output, _ = self.lstm(packed_input, (h_0, c_0))
        # Unpack the output sequence
        # output shape: (batch_size, max_seq_len, lstm_hidden_dim)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)
        
        # Pass LSTM outputs through the final fully connected layer
        # reconstructed_x shape: (batch_size, max_seq_len, output_feature_dim)
        reconstructed_x = self.fc_out(output)
        return reconstructed_x

    def generate_from_latent(self, z, max_len, device, temperature=1.0):
        # THIS METHOD IS FOR AUTOREGRESSIVE GENERATION
        batch_size = z.size(0)
        
        # 1. Project latent vector z to initial hidden and cell states
        initial_hidden_flat = self.latent_to_hidden(z)
        initial_cell_flat = self.latent_to_cell(z)
        
        # Reshape for LSTM: (num_layers, batch_size, lstm_hidden_dim)
        h_t = initial_hidden_flat.view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()
        c_t = initial_cell_flat.view(batch_size, self.num_layers, self.lstm_hidden_dim).permute(1, 0, 2).contiguous()
        
        # 2. Initialize the first input step (start token) as a tensor of zeros
        # Shape: (batch_size, 1, rnn_input_dim)
        input_t = torch.zeros(batch_size, 1, self.lstm_input_dim).to(device)
        
        outputs = []
        # 3. Autoregressive loop
        for _ in range(max_len):
            # Pass current input and hidden state to LSTM
            # output_t shape: (batch_size, 1, lstm_hidden_dim)
            # h_t, c_t are updated for the next step
            output_t, (h_t, c_t) = self.lstm(input_t, (h_t, c_t))
            
            # Pass LSTM output to the final linear layer
            # prediction_logits shape: (batch_size, 1, output_dim)
            prediction_logits = self.fc_out(output_t)

            # Apply temperature and sigmoid to get the next step's input
            # This turns the logits into a probability-like pianoroll slice
            # next_input = torch.sigmoid(prediction_logits / temperature)
            
            # Store the prediction for this step (after activation)
            outputs.append(prediction_logits)
            
            # Set the input for the next time step
            input_t = prediction_logits

        # Concatenate all the generated steps along the time dimension (dim=1)
        generated_sequence = torch.cat(outputs, dim=1)
        
        return generated_sequence