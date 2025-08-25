import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from config import *
from dataset import MidiDataset
from utils import pianoroll_tensor_to_midi

class ImprovedEncoder(nn.Module):
    def __init__(self, hidden_dim, z_dim, n_layers, dropout=0.2):
        super(ImprovedEncoder, self).__init__()
        
        # Use unidirectional LSTM to match decoder complexity
        self.lstm = nn.LSTM(
            NUM_PITCHES, 
            hidden_dim, 
            n_layers, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False  # Changed to unidirectional
        )
        
        # Add batch normalization and dropout for regularization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Linear layers for mean and variance
        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        
    def forward(self, x, lengths):
        # x shape: (batch, pitches, time) -> (batch, time, pitches)
        x = x.permute(0, 2, 1)
        
        # Pack sequence for efficiency
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # Get LSTM output and final hidden state
        packed_output, (hidden, _) = self.lstm(packed_x)
        
        # Use the last hidden state from the top layer
        # hidden shape: (n_layers, batch, hidden_dim)
        final_hidden = hidden[-1]  # Take last layer: (batch, hidden_dim)
        
        # Apply batch norm and dropout
        final_hidden = self.batch_norm(final_hidden)
        final_hidden = self.dropout(final_hidden)
        
        # Get mean and log variance
        mean = self.fc_mean(final_hidden)
        logvar = self.fc_logvar(final_hidden)
        
        return mean, logvar

class ImprovedDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, n_layers, dropout=0.2):
        super(ImprovedDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Multi-layer LSTM decoder
        self.lstm = nn.LSTM(
            NUM_PITCHES, 
            hidden_dim, 
            n_layers, 
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Project latent to initial hidden and cell states
        self.fc_z_to_hidden = nn.Linear(z_dim, n_layers * hidden_dim)
        self.fc_z_to_cell = nn.Linear(z_dim, n_layers * hidden_dim)
        
        # Output projection with residual connection
        self.fc_out = nn.Linear(hidden_dim, NUM_PITCHES)
        self.fc_residual = nn.Linear(NUM_PITCHES, NUM_PITCHES)
        
        # Batch normalization for output
        self.batch_norm_out = nn.BatchNorm1d(NUM_PITCHES)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, z, max_length):
        batch_size = x.size(0)
        
        # Initialize hidden and cell states from latent vector
        hidden = self.fc_z_to_hidden(z).view(
            batch_size, self.n_layers, self.hidden_dim
        ).transpose(0, 1).contiguous()
        
        cell = self.fc_z_to_cell(z).view(
            batch_size, self.n_layers, self.hidden_dim
        ).transpose(0, 1).contiguous()
        
        # Prepare input sequence (shifted by one timestep)
        decoder_input = x.permute(0, 2, 1)  # (batch, time, pitches)
        
        # Add start token (zeros)
        start_token = torch.zeros(batch_size, 1, NUM_PITCHES).to(x.device)
        decoder_input = torch.cat([start_token, decoder_input[:, :-1, :]], dim=1)
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(decoder_input, (hidden, cell))
        
        # Project to output space with residual connection
        output = self.fc_out(lstm_output)
        residual = self.fc_residual(decoder_input)
        output = output + 0.1 * residual  # Small residual weight
        
        # Apply batch norm (reshape for 1D batch norm)
        output_reshaped = output.permute(0, 2, 1)  # (batch, pitches, time)
        
        return output_reshaped

class LofiModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, n_layers=2, dropout=0.2):
        super(LofiModel, self).__init__()
        
        self.encoder = ImprovedEncoder(hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = ImprovedDecoder(latent_dim, hidden_dim, n_layers, dropout)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Xavier initialization for better training stability"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick with optional β-VAE scaling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, lengths, teacher_forcing_ratio=1.0, beta=1.0):
        # Encode
        mean, logvar = self.encoder(x, lengths)
        
        # Reparameterize
        z = self.reparameterize(mean, logvar)
        
        # Decode
        max_len = int(lengths.max().item())
        reconstruction = self.decoder(x, z, max_len)
        
        # Ensure output matches input dimensions
        if reconstruction.shape[2] < x.shape[2]:
            padding_needed = x.shape[2] - reconstruction.shape[2]
            reconstruction = F.pad(reconstruction, (0, padding_needed))
        elif reconstruction.shape[2] > x.shape[2]:
            reconstruction = reconstruction[:, :, :x.shape[2]]
        
        return reconstruction, mean, logvar
    
    def generate(self, num_samples=1, max_len=MAX_SEQ_LEN, visualize=True, save_path=None, temperature=0.8, 
                 threshold=0.01, device=None):
        """Generate new samples from the latent space"""
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Create dummy input for decoder
            dummy_input = torch.zeros(num_samples, NUM_PITCHES, max_len).to(device)
            
            # Generate
            generated = self.decoder(dummy_input, z, max_len)
            
            # Apply temperature scaling and threshold
            generated = torch.sigmoid(generated / temperature)
            # generated[generated < threshold] = 0
        
        generated_sequence = generated.squeeze(0).cpu()
        generated_sequence[generated_sequence < threshold] = 0
        if visualize:
            MidiDataset.visualize_midi(generated_sequence)
        if save_path:
            pianoroll_tensor_to_midi(generated_sequence, save_path)
        return generated_sequence
            
    
    def reconstruct(self, x, lengths, visualize=True):
        """Reconstruct input sequences"""
        random_tensor_batch = x.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        random_length_batch = torch.tensor([lengths], dtype=torch.long)
        
        recon_logits, _, _ = self(random_tensor_batch, random_length_batch, 0.0, beta=1.0)
        
        recon_tensor = torch.sigmoid(recon_logits).squeeze(0).detach().cpu()
        
        if visualize:
            MidiDataset.visualize_midi(recon_tensor)