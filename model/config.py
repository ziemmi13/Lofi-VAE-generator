import pretty_midi

# Training variables
TRAIN_VALIDATION_SPLIT = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 200

# Model parameters
LATENT_DIM = 128     
HIDDEN_DIM = 256     
LSTM_LAYERS = 1   

# Loss parameters
KLD_MAX_WEIGHT = 0.02
KLD_WARMUP_EPOCHS = 40

# --- NEW: Scheduled Sampling Parameter ---
# Start with 100% teacher forcing and slowly decay
TEACHER_FORCING_RATIO = 1.0 

FS= 10

# DATASET
MIN_MIDI_NOTE = pretty_midi.note_name_to_number('A0')
MAX_MIDI_NOTE = pretty_midi.note_name_to_number('C7') 
NUM_PITCHES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1
INPUT_DIM = NUM_PITCHES
MAX_SEQ_LEN = 300 # 30 s