import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from config import *
from dataset import MidiDataset
from utils import pianoroll_tensor_to_midi

# --- Encoder and Decoder classes remain exactly the same ---
class Encoder(nn.Module):
    def __init__(self, hidden_dim, z_dim, n_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(NUM_PITCHES, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.fc_mean = nn.Linear(2 * hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, z_dim)
    def forward(self, x, lengths):
        x = x.permute(0, 2, 1) 
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hidden, _) = self.lstm(packed_x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, n_layers):
        super(Decoder, self).__init__()
        # Use LSTMCell for step-by-step decoding, which is more explicit for scheduled sampling.
        self.lstm_cell = nn.LSTMCell(NUM_PITCHES, hidden_dim)
        self.fc = nn.Linear(hidden_dim, NUM_PITCHES)
        self.n_layers = n_layers # Assuming n_layers=1 for this simplified LSTMCell example
        self.hidden_dim = hidden_dim
    
    # Note: A multi-layer LSTMCell is more complex to implement. For this example, we'll
    # simplify and assume n_layers=1 in the decoder for clarity. If you need multi-layer,
    # you would stack LSTMCells in a loop here.
    def forward(self, x, hidden, cell):
        # x shape: (batch_size, num_pitches)
        hidden, cell = self.lstm_cell(x, (hidden, cell))
        output_logits = self.fc(hidden)
        return output_logits, hidden, cell

class LofiModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, n_layers=LSTM_LAYERS):
        super(LofiModel, self).__init__()
        self.encoder = Encoder(hidden_dim, latent_dim, n_layers)
        # Note: The decoder is simplified to a single layer for this implementation.
        # To match LSTM_LAYERS > 1, you would need a more complex decoder.
        self.decoder = Decoder(latent_dim, hidden_dim, 1) # Using 1 layer for LSTMCell
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_dim) # For single layer decoder
        self.fc_latent_to_cell = nn.Linear(latent_dim, hidden_dim)   # For single layer decoder
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, lengths, teacher_forcing_ratio=1.0):
        """
        The main forward pass, now with a step-by-step loop for Scheduled Sampling.
        """
        mean, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mean, logvar)
        
        batch_size = x.size(0)
        # Project z to the initial hidden/cell states for the single-layer decoder
        hidden = self.fc_latent_to_hidden(z)
        cell = self.fc_latent_to_cell(z)
        
        # The maximum sequence length in this specific batch
        max_len = int(lengths.max().item())
        
        # Start with a "start of sequence" token
        decoder_input = torch.zeros(batch_size, NUM_PITCHES).to(self.device)
        
        # Prepare ground truth for teacher forcing
        ground_truth_sequence = x.permute(0, 2, 1)

        reconstruction_logits_list = []

        for t in range(max_len):
            output_logits, hidden, cell = self.decoder(decoder_input, hidden, cell)
            reconstruction_logits_list.append(output_logits)

            # Decide whether to use teacher forcing for the next step
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Use the ground-truth next step as the next input
                decoder_input = ground_truth_sequence[:, t, :]
            else:
                # Use the model's own output as the next input
                output_probs = torch.sigmoid(output_logits)
                # For even better results, sample from the distribution
                decoder_input = torch.bernoulli(output_probs)
        
        reconstruction_logits = torch.stack(reconstruction_logits_list, dim=1)
        reconstruction_logits = reconstruction_logits.permute(0, 2, 1)

        # Pad the output to match the original padded input shape `x`
        if reconstruction_logits.shape[2] < x.shape[2]:
            padding_needed = x.shape[2] - reconstruction_logits.shape[2]
            reconstruction_logits = F.pad(reconstruction_logits, (0, padding_needed))

        return reconstruction_logits, mean, logvar
    
    def generate(self, max_len=MAX_SEQ_LEN, visualize=True, threshold=0.01, save_path=None, temperature=0.8):
        self.eval()
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(self.device)
            # Project z to initial hidden/cell states for the single-layer decoder
            hidden = self.fc_latent_to_hidden(z)
            cell = self.fc_latent_to_cell(z)
            
            decoder_input = torch.zeros(1, NUM_PITCHES).to(self.device)
            
            generated_sequence_logits = []
            for _ in range(max_len):
                output_logits, hidden, cell = self.decoder(decoder_input, hidden, cell)
                
                scaled_logits = output_logits / temperature
                output_probs = torch.sigmoid(scaled_logits)
                
                # Sample from the probabilities to get a concrete on/off decision
                decoder_input = torch.bernoulli(output_probs)
                
                generated_sequence_logits.append(output_logits)

            generated_sequence = torch.stack(generated_sequence_logits, dim=1)
            generated_sequence = torch.sigmoid(generated_sequence)
            generated_sequence = generated_sequence.permute(0, 2, 1)
            
            generated_sequence = generated_sequence.squeeze(0).cpu()
            generated_sequence[generated_sequence < threshold] = 0
            if visualize:
                MidiDataset.visualize_midi(generated_sequence)
            if save_path:
                pianoroll_tensor_to_midi(generated_sequence, save_path)
            return generated_sequence

    # reconstruct method would need to be updated similarly if used
    def reconstruct(self, x, lengths):
        self.eval()
        lengths_tensor = torch.tensor([lengths], dtype=torch.long)
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            reconstructed_logits, _, _ = self(x, lengths_tensor, 0.0) # No teacher forcing
        reconstructed_x = torch.sigmoid(reconstructed_logits)
        reconstructed_x[reconstructed_x < 0.01] = 0
        reconstructed_x = reconstructed_x.squeeze(0).cpu()
        MidiDataset.visualize_midi(reconstructed_x)

        return reconstructed_x
