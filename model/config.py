# Training variables
TRAIN_VALIDATION_SPLIT = 0.85
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 500
LATENT_DIM = 256
LSTM_LAYERS = 2

# 
# MIDI_LEN = 256
# PIANOROLL_RANGE = 88
# FS=32
FS=8

# DATASET
MIN_MIDI_NOTE = 21  # A0 (MIDI note number). This is the lowest MIDI note we will consider.
MAX_MIDI_NOTE = 108 # C8 (MIDI note number). This is the highest MIDI note we will consider.
# The number of unique notes in our piano roll representation.
# For 88 keys: 108 - 21 + 1 = 88. This is our feature dimension for each time step.
INPUT_DIM = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1 
NUM_PITCHES = INPUT_DIM

# Dataset Filtering
MIN_SEQ_LEN_FILTER = 10  # Minimum number of time steps for a sequence to be included.
MAX_SEQ_LEN_FILTER = 5000 # Maximum number of time steps for a sequence to be included.


NUM_INSTRUMENTS = 4
