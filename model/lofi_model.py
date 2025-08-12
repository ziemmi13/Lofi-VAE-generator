# model.py
import torch
import torch.nn as nn
from config import *
from dataset import MidiDataset
from utils import pianoroll_tensor_to_midi

class Encoder(nn.Module):
    def __init__(self, hidden_dim, z_dim, n_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            NUM_PITCHES,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True
        )
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
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            NUM_PITCHES,
            hidden_dim,
            n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, NUM_PITCHES)

    def forward(self, x, hidden, cell):
        # This forward pass is now designed to be called one step at a time.
        lstm_out, (hidden, cell) = self.lstm(x, (hidden, cell))
        output_logits = self.fc(lstm_out)
        return output_logits, hidden, cell

class LofiModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, n_layers=LSTM_LAYERS):
        super(LofiModel, self).__init__()
        self.encoder = Encoder(hidden_dim, latent_dim, n_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, n_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc_latent_to_hidden = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.fc_latent_to_cell = nn.Linear(latent_dim, n_layers * hidden_dim)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, lengths):
        mean, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mean, logvar)
        
        hidden_init_flat = self.fc_latent_to_hidden(z)
        cell_init_flat = self.fc_latent_to_cell(z)
        batch_size = x.size(0)
        decoder_hidden_init = hidden_init_flat.view(self.n_layers, batch_size, self.hidden_dim)
        decoder_cell_init = cell_init_flat.view(self.n_layers, batch_size, self.hidden_dim)
        
        decoder_input = x.permute(0, 2, 1)
        
        # For efficiency, PyTorch's LSTM can process the whole sequence for teacher forcing.
        lstm_out, _ = self.decoder.lstm(decoder_input, (decoder_hidden_init, decoder_cell_init))
        reconstruction_logits = self.decoder.fc(lstm_out)
        reconstruction_logits = reconstruction_logits.permute(0, 2, 1)

        # *** CRITICAL FIX ***
        # Apply sigmoid to scale the output to [0, 1], matching the normalized input data.
        reconstruction = torch.sigmoid(reconstruction_logits)

        return reconstruction, mean, logvar
    
    def reconstruct(self, x, lengths):
        self.eval()
        lengths_tensor = torch.tensor([lengths], dtype=torch.long)
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            reconstructed_x, _, _ = self(x, lengths_tensor)
        reconstructed_x = reconstructed_x.squeeze(0).cpu()
        MidiDataset.visualize_midi(reconstructed_x)
    
    def generate(self, max_len=MAX_SEQ_LEN, visualize=True, threshold=0.01, save_path=None):
        self.eval()
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(self.device)
            hidden_init_flat = self.fc_latent_to_hidden(z)
            cell_init_flat = self.fc_latent_to_cell(z)
            hidden = hidden_init_flat.view(self.n_layers, 1, self.hidden_dim)
            cell = cell_init_flat.view(self.n_layers, 1, self.hidden_dim)
            decoder_input = torch.zeros(1, 1, NUM_PITCHES).to(self.device)

            generated_sequence_logits = []
            for _ in range(max_len):
                # Call the decoder one step at a time
                output_logits, hidden, cell = self.decoder(decoder_input, hidden, cell)
                
                # Apply sigmoid to get probabilities in [0, 1] range.
                output_probs = torch.sigmoid(output_logits)

                # The ACTIVATED output becomes the input for the next time step.
                decoder_input = output_probs
                
                generated_sequence_logits.append(output_logits.squeeze(1))

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
