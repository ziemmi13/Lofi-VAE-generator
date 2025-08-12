import pretty_midi

# Training variables
TRAIN_VALIDATION_SPLIT = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # A slightly lower learning rate can improve stability
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200

# Model parameters
LATENT_DIM = 64      
HIDDEN_DIM = 256     # A powerful decoder is less likely to ignore the latent space
LSTM_LAYERS = 2      

# Loss parameters for stable training
KLD_MAX_WEIGHT = 0.02        # The target KL weight
KLD_WARMUP_EPOCHS = 40      # CRITICAL: Number of epochs with ZERO KL weight

FS= 10

# DATASET
MIN_MIDI_NOTE = pretty_midi.note_name_to_number('A0')
MAX_MIDI_NOTE = pretty_midi.note_name_to_number('C7') 
NUM_PITCHES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1
INPUT_DIM = NUM_PITCHES
MAX_SEQ_LEN = 300 # 30 s
