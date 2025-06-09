# Training variables
TRAIN_VALIDATION_SPLIT = 0.9
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 500
LATENT_DIM = 256
LSTM_LAYERS = 2

FS=8

# DATASET
MIN_MIDI_NOTE = 21  # A0 (MIDI note number). This is the lowest MIDI note we will consider.
MAX_MIDI_NOTE = 108 # C8 (MIDI note number). This is the highest MIDI note we will consider.
# The number of unique notes in our piano roll representation.
# For 88 keys: 108 - 21 + 1 = 88. This is our feature dimension for each time step.
 
NUM_PITCHES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1
NUM_INSTRUMENTS = 4
INPUT_DIM = NUM_INSTRUMENTS * NUM_PITCHES

# Dataset Filtering
MIN_SEQ_LEN_FILTER = 10  # Minimum number of time steps for a sequence to be included.
MAX_SEQ_LEN_FILTER = 5000 # Maximum number of time steps for a sequence to be included.


