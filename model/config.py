# Training variables
TRAIN_VALIDATION_SPLIT = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 5
LATENT_DIM = 256
LSTM_LAYERS = 2

FS=32

# DATASET
MIN_MIDI_NOTE = 0  # A0 (MIDI note number). This is the lowest MIDI note we will consider.
MAX_MIDI_NOTE = 128 # C8 (MIDI note number). This is the highest MIDI note we will consider.
# The number of unique notes in our piano roll representation.
# For 88 keys: 108 - 21 + 1 = 88. This is our feature dimension for each time step.
 
# NUM_PITCHES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1
NUM_PITCHES = 128
NUM_INSTRUMENTS = 5
INPUT_DIM = NUM_INSTRUMENTS * NUM_PITCHES

MAX_SEQ_LEN = 1920 # ONE MINUTE


