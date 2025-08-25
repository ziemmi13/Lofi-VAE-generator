import pretty_midi

# Training variables
TRAIN_VALIDATION_SPLIT = 0.8
BATCH_SIZE = 32  # Reduced for better gradient estimates
LEARNING_RATE = 1e-4  # Reduced learning rate
WEIGHT_DECAY = 1e-4   # Increased weight decay
NUM_EPOCHS = 300      # More epochs with lower LR

# Model parameters
LATENT_DIM = 64       # Reduced latent dimension
HIDDEN_DIM = 512      # Increased hidden dimension
LSTM_LAYERS = 2       # Multi-layer architecture

# Loss parameters - CRITICAL CHANGES
KLD_MAX_WEIGHT = 1.0  # Much higher KL weight
KLD_WARMUP_EPOCHS = 150  # Longer warmup
BETA_VAE_BETA = 1.5   # β-VAE coefficient for better disentanglement

# Teacher forcing schedule - IMPROVED
TEACHER_FORCING_RATIO = 1.0
TEACHER_FORCING_MIN = 0.3     # Don't decay below this
TEACHER_FORCING_DECAY = 0.999  # Much slower decay

# Data parameters
FS = 10
MIN_MIDI_NOTE = pretty_midi.note_name_to_number('A0')
MAX_MIDI_NOTE = pretty_midi.note_name_to_number('C7')
NUM_PITCHES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1
INPUT_DIM = NUM_PITCHES
MAX_SEQ_LEN = 200

# Regularization
DROPOUT_RATE = 0.3
GRADIENT_CLIP_NORM = 1.0

# Loss weights for better balance
RECONSTRUCTION_WEIGHT = 1.0
KL_WEIGHT_SCHEDULE = "cosine"  # or "linear", "exponential"