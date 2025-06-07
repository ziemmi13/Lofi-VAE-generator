import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import *
from dataset import MidiDataset
import pretty_midi

class LofiModel(nn.Module):
    def __init__(self, device):
        super(LofiModel, self).__init__()
        self.device = device
        # Decoder's input_dim for LSTM should match INPUT_DIM for teacher forcing
        self.encoder = Encoder(input_dim=INPUT_DIM, hidden_dim=LATENT_DIM, latent_dim=LATENT_DIM, num_layers=LSTM_LAYERS)
        self.decoder = Decoder(rnn_input_dim=INPUT_DIM, # For teacher forcing, LSTM input is from x (INPUT_DIM)
                               lstm_hidden_dim=LATENT_DIM,
                               z_latent_dim=LATENT_DIM,
                               output_dim=INPUT_DIM,
                               num_layers=LSTM_LAYERS)

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            logvar (torch.Tensor): Log variance of the latent Gaussian.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        z = mu + eps * std
        return z
        
    def forward(self, x, lengths):
        """
        Performs a forward pass of VAE  
        Args:
            x (torch.Tensor): Input MIDI (batch_size, max_seq_len, input_dim).
            lengths (torch.Tensor): Original lengths of sequences in the batch.

        Returns:
            reconstructed_x (torch.Tensor): Reconstructed input (batch_size, max_seq_len, input_dim).
            mu (torch.Tensor): Latent mean (batch_size, latent_dim).
            logvar (torch.Tensor): Latent log variance (batch_size, latent_dim).
        """
        
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        # For teacher forcing, decoder uses original x as input sequence
        reconstructed_x = self.decoder(x, z, lengths)

        return reconstructed_x, mu, logvar

    def reconstruct(self, midi_tensor, save_to_midi=True, save_path="reconstructions/reconstructed.mid"):
        # Prepare tensor
        tensor_len = midi_tensor.shape[0]
        padded_tensor, tensor_len = MidiDataset.collate_fn([(midi_tensor, tensor_len)])
        padded_tensor = padded_tensor.to(self.device)

        # Reconstruct sample
        reconstructed_sample, _, _ = self(padded_tensor, tensor_len)
        reconstructed_sample = reconstructed_sample.squeeze()
        reconstructed_sample = torch.sigmoid(reconstructed_sample)

        #Threshold
        reconstructed_sample[reconstructed_sample < 0.5] = 0
        reconstructed_sample[reconstructed_sample >= 0.5] = 1

        # Transpose to convert into midi
        reconstructed_sample_T = reconstructed_sample.T

        if save_to_midi:
            # Create a PrettyMIDI object
            midi = pretty_midi.PrettyMIDI()
            piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
            piano = pretty_midi.Instrument(program=piano_program)

            # Track note on/off times per pitch
            # We'll detect note start and end by scanning through time steps
            for pitch_idx in range(reconstructed_sample_T.shape[0]):
                note_on = None
                for t in range(reconstructed_sample_T.shape[1]):
                    if reconstructed_sample_T[pitch_idx, t] == 1 and note_on is None:
                        # Note on at time t/fs seconds
                        note_on = t / FS
                    elif (reconstructed_sample_T[pitch_idx, t] == 0 or t == reconstructed_sample_T.shape[0]-1) and note_on is not None:
                        # Note off at time t/fs seconds
                        note_off = t / FS
                        # Add the note to the instrument
                        note = pretty_midi.Note(    
                            velocity=50,
                            pitch= pitch_idx+MIN_MIDI_NOTE,
                            start=note_on,
                            end=note_off
                        )
                        piano.notes.append(note)
                        note_on = None  # reset for next note

                # If a note is still on at the end, close it
                if note_on is not None:
                    note_off = reconstructed_sample_T.shape[1] / FS
                    note = pretty_midi.Note(
                        velocity=50,
                        pitch= pitch_idx+MIN_MIDI_NOTE,
                        start=note_on,
                        end=note_off
                    )
                    piano.notes.append(note)

            # Add instrument to the PrettyMIDI object
            midi.instruments.append(piano)
        
        # Write out the MIDI data
        midi.write(save_path)



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
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=LATENT_DIM, latent_dim=LATENT_DIM, num_layers=LSTM_LAYERS, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim # LSTM hidden size
        self.num_layers = num_layers
        self.latent_dim = latent_dim # VAE latent z dimension

        self.lstm = nn.LSTM(input_size=input_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, # Corrected: use self.num_layers
                              batch_first=True,
                              dropout=dropout if self.num_layers > 1 else 0.0) # Corrected: use dropout param

        # Project LSTM's final hidden state to mean and logvar of latent distribution
        # Input: (batch, num_layers * hidden_dim) -> Output: (batch, latent_dim)
        self.hidden_to_mu = nn.Linear(self.hidden_dim * self.num_layers, self.latent_dim)
        self.hidden_to_logvar = nn.Linear(self.hidden_dim * self.num_layers, self.latent_dim)

    def forward(self, x, lengths):
        """
        Forward pass for the encoder.
        Args:
            x (torch.Tensor): Padded input sequences (batch_size, max_seq_len, input_dim).
            lengths (torch.Tensor): Original lengths of sequences in the batch.
        Returns:
            tuple: mu and logvar of the latent space (each shape: batch_size, latent_dim)
        """
        # Pack padded batch of sequences for the LSTM
        # Ensure lengths are on CPU and are integers/long for pack_padded_sequence
        packed_input = pack_padded_sequence(x, lengths.cpu().long(), batch_first=True, enforce_sorted=False)

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
    def __init__(self, rnn_input_dim=INPUT_DIM, lstm_hidden_dim=LATENT_DIM, 
                 z_latent_dim=LATENT_DIM, output_dim=INPUT_DIM, 
                 num_layers=LSTM_LAYERS, dropout=0.1):
        super(Decoder, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.z_latent_dim = z_latent_dim
        self.output_feature_dim = output_dim

        # Project latent vector z to initial hidden and cell states for the LSTM
        # Input: z_latent_dim -> Output: num_layers * lstm_hidden_dim
        self.latent_to_hidden = nn.Linear(self.z_latent_dim, self.num_layers * self.lstm_hidden_dim)
        self.latent_to_cell = nn.Linear(self.z_latent_dim, self.num_layers * self.lstm_hidden_dim)

        # LSTM:
        # input_size: dimension of input features at each time step (from x for teacher forcing)
        # hidden_size: dimension of LSTM hidden units
        self.lstm = nn.LSTM(input_size=rnn_input_dim, # Corrected: This is the dim of teacher-forced input x_t
                              hidden_size=self.lstm_hidden_dim,
                              num_layers=self.num_layers, # Corrected: use self.num_layers
                              batch_first=True,
                              dropout=dropout if self.num_layers > 1 else 0.0) # Corrected: use dropout param

        # Output layer: maps LSTM hidden state to output feature dimension
        self.fc_out = nn.Linear(self.lstm_hidden_dim, self.output_feature_dim)
        
        # If your output features are probabilities (e.g., for a piano roll),
        # you might want a sigmoid activation here or after the loss calculation.
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x_teacher_force, z, lengths):
        """
        Forward pass for the decoder.
        Args:
            x_teacher_force (torch.Tensor): Ground truth sequences for teacher forcing
                                           (batch_size, max_seq_len, rnn_input_dim).
            z (torch.Tensor): Latent vector (batch_size, z_latent_dim).
            lengths (torch.Tensor): Original lengths of sequences in the batch.
        Returns:
            torch.Tensor: Reconstructed sequences (batch_size, max_seq_len, output_feature_dim).
        """
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
        packed_input = pack_padded_sequence(x_teacher_force, lengths.cpu().long(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, _ = self.lstm(packed_input, (h_0, c_0))
        
        # Unpack the output sequence
        # output shape: (batch_size, max_seq_len, lstm_hidden_dim)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)
        
        # Pass LSTM outputs through the final fully connected layer
        # reconstructed_x shape: (batch_size, max_seq_len, output_feature_dim)
        reconstructed_x = self.fc_out(output)

        # if self.sigmoid:
        #     reconstructed_x = self.sigmoid(reconstructed_x)
            
        return reconstructed_x