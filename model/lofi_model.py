import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pretty_midi
from config import *

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
        self.decoder = Decoder(rnn_input_dim=self.input_dim,
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

    def reconstruct(self, midi_tensor, save_to_midi=True, save_path="reconstructions/reconstructed.mid"):
        # Prepare tensor
        tensor_len = midi_tensor.shape[0]
        padded_tensor, tensor_len = MidiDataset.collate_fn([(midi_tensor, tensor_len)])
        padded_tensor = padded_tensor.to(self.device)

        # Reconstruct sample
        reconstructed_sample, _, _ = self(padded_tensor, tensor_len)
        reconstructed_sample = reconstructed_sample.squeeze()

        reconstructed_sample = torch.clamp(reconstructed_sample, 0.0, 1.0)

        #Threshold
        reconstructed_sample[reconstructed_sample < 0.01] = 0.0

        # Transpose to convert into midi
        reconstructed_sample_T = reconstructed_sample.T

        # Scale back to original velocity
        # reconstructed_sample_T *= 127


        midi = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        instrument = pretty_midi.Instrument(program=midi)

        for pitch_idx in range(reconstructed_sample_T.shape[0]):
            note_on_time = None
            peak_velocity_normalized = 0

            for t in range(reconstructed_sample_T.shape[1]):
                current_velocity_normalized = reconstructed_sample_T[pitch_idx, t].item()
                is_note_on = current_velocity_normalized > 0.1

                # --- Note On Event ---
                if is_note_on and note_on_time is None:
                    note_on_time = t / FS
                    peak_velocity_normalized = current_velocity_normalized
                
                # --- Note Continues ---
                elif is_note_on and note_on_time is not None:
                    # Update the peak velocity if the current one is higher
                    if current_velocity_normalized > peak_velocity_normalized:
                        peak_velocity_normalized = current_velocity_normalized

                # --- Note Off Event ---
                elif not is_note_on and note_on_time is not None:
                    note_off_time = t / FS
                    
                    # Un-normalize velocity to MIDI range [0, 127]
                    velocity_midi = int(peak_velocity_normalized * 127)
                    if velocity_midi > 127: velocity_midi = 127
                    if velocity_midi == 0: velocity_midi = 1 # Velocity 0 is note-off

                    note = pretty_midi.Note(
                        velocity=velocity_midi,
                        pitch=pitch_idx + MIN_MIDI_NOTE,
                        start=note_on_time,
                        end=note_off_time
                    )
                    instrument.notes.append(note)
                    
                    # Reset for the next note
                    note_on_time = None
                    peak_velocity_normalized = 0

            # If a note is still on at the very end of the sequence, close it
            if note_on_time is not None:
                note_off_time = reconstructed_sample_T.shape[1] / FS
                velocity_midi = int(peak_velocity_normalized * 127)
                if velocity_midi > 127: velocity_midi = 127
                if velocity_midi == 0: velocity_midi = 1

                note = pretty_midi.Note(
                    velocity=velocity_midi,
                    pitch=pitch_idx + MIN_MIDI_NOTE,
                    start=note_on_time,
                    end=note_off_time
                )
                instrument.notes.append(note)

        midi.instruments.append(instrument)
        midi.write(save_path)
        print(f"Reconstructed MIDI saved to {save_path}")


    def generate(self, num_samples, max_len, z_sample=None, temperature=1.0):
        """
        Generates new sequences from the model.

        Args:
            num_samples (int): Number of sequences to generate.
            max_len (int): Maximum length of the generated sequences.
            z_sample (torch.Tensor, optional): A specific latent vector (or batch of vectors)
                                               to decode. If None, samples from prior.
                                               Shape: (num_samples, latent_dim) or (latent_dim).
            temperature (float): Softmax temperature for sampling (if output is categorical).
                                 Not directly used in this continuous output example, but
                                 good to keep in mind for variants. For continuous outputs,
                                 it could scale noise if you were adding any.

        Returns:
            torch.Tensor: Generated sequences (num_samples, max_len, output_feature_dim).
        """
        self.eval() 
        with torch.no_grad(): 
            if z_sample is None:
                # Sample z from the prior distribution (standard Gaussian)
                z = torch.randn(num_samples, LATENT_DIM).to(self.device)
            else:
                z = z_sample.to(self.device)
                if z.ndim == 1: # If a single z vector is provided for one sample
                    z = z.unsqueeze(0)
                if z.shape[0] != num_samples:
                    print(f"Warning: num_samples ({num_samples}) does not match z_sample batch size ({z.shape[0]}). Using z_sample batch size.")
                    num_samples = z.shape[0]
                if z.shape[1] != self.decoder.z_latent_dim:
                    raise ValueError(f"Provided z_sample has latent_dim {z.shape[1]}, expected {self.decoder.z_latent_dim}")


            # Project latent vector z to initial hidden and cell states for the LSTM
            # h_flat/c_flat shape: (num_samples, num_layers * lstm_hidden_dim)
            initial_hidden_flat = self.decoder.latent_to_hidden(z)
            initial_cell_flat = self.decoder.latent_to_cell(z)

            # Reshape to (num_layers, num_samples, lstm_hidden_dim) for LSTM
            h_t = initial_hidden_flat.view(num_samples, self.decoder.num_layers, self.decoder.lstm_hidden_dim).permute(1, 0, 2).contiguous()
            c_t = initial_cell_flat.view(num_samples, self.decoder.num_layers, self.decoder.lstm_hidden_dim).permute(1, 0, 2).contiguous()

            # Initial input token for the LSTM.
            # A common choice is a zero vector.
            # Shape: (num_samples, 1, rnn_input_dim)
            # self.decoder.lstm.input_size is rnn_input_dim (which is INPUT_DIM)
            current_input_token = torch.zeros(num_samples, 1, self.decoder.lstm.input_size).to(self.device)

            generated_sequence_parts = []

            for _ in range(max_len):
                # LSTM forward pass for one step
                # lstm_out shape: (num_samples, 1, lstm_hidden_dim)
                # h_t, c_t will be updated to the next states
                lstm_out, (h_t, c_t) = self.decoder.lstm(current_input_token, (h_t, c_t))

                # Pass LSTM output through the final fully connected layer
                # lstm_out.squeeze(1) shape: (num_samples, lstm_hidden_dim)
                # output_token_features shape: (num_samples, output_feature_dim)
                output_token_features = self.decoder.fc_out(lstm_out.squeeze(1))

                # --- Post-processing of output_token_features ---
                # If your output represents probabilities (e.g., for a piano roll),
                # you should apply a sigmoid activation.
                # If your model was trained with nn.BCEWithLogitsLoss, then fc_out produces logits,
                # and you should apply sigmoid here.
                # If your model was trained with nn.MSELoss on data already in [0,1],
                # a sigmoid here might still be beneficial to ensure output is in range.
                output_token_processed = torch.sigmoid(output_token_features)
                # output_token_processed = output_token_features # If no sigmoid is desired / already handled

                generated_sequence_parts.append(output_token_processed.unsqueeze(1)) # Shape: (num_samples, 1, output_feature_dim)

                # The next input to the LSTM is the output we just generated.
                # It must have the shape (num_samples, 1, rnn_input_dim)
                # This assumes self.decoder.output_feature_dim == self.decoder.lstm.input_size (INPUT_DIM)
                current_input_token = output_token_processed.unsqueeze(1)

            # Concatenate all generated steps along the sequence length dimension
            # Result shape: (num_samples, max_len, output_feature_dim)
            final_generated_sequence = torch.cat(generated_sequence_parts, dim=1)

        # self.train() # Optional: set back to train mode if you plan to continue training
        return final_generated_sequence


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
    def __init__(self, rnn_input_dim, lstm_hidden_dim, z_latent_dim, output_dim, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.latent_to_hidden = nn.Linear(z_latent_dim, num_layers * lstm_hidden_dim)
        self.latent_to_cell = nn.Linear(z_latent_dim, num_layers * lstm_hidden_dim)
        self.lstm = nn.LSTM(input_size=rnn_input_dim,
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