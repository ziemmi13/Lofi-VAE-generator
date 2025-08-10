import pretty_midi

# Training variables
TRAIN_VALIDATION_SPLIT = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
LATENT_DIM = 64
HIDDEN_DIM = 64
LSTM_LAYERS = 2

FS= 10

# DATASET
MIN_MIDI_NOTE = pretty_midi.note_name_to_number('A0') # C2 (MIDI note number). This is the lowest MIDI note we will consider.
MAX_MIDI_NOTE = pretty_midi.note_name_to_number('C7') 
# The number of unique notes in our piano roll representation.
# For 88 keys: 108 - 21 + 1 = 88. This is our feature dimension for each time step.
 
NUM_PITCHES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1
# NUM_PITCHES = 128
INPUT_DIM = NUM_PITCHES

MAX_SEQ_LEN = 300 # 1 min

