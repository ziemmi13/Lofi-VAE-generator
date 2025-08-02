from config import *
import numpy as np

def drum_to_pianoroll(instrument):
    """
    Create a pianoroll for a drum track manually because for instrument.is_drum 
    the function: instrument.get_pianoroll() doesn't work.
    """
    end_time = max(note.end for note in instrument.notes)
    n_frames = int(end_time * FS) + 1
    pianoroll = np.zeros((NUM_PITCHES, n_frames)) 

    for note in instrument.notes:
        start = int(note.start * FS)
        end = int(note.end * FS)
        pitch = note.pitch
        velocity = note.velocity
        
        # Fill values in the piano roll
        pianoroll[pitch, start:end] = velocity

    return pianoroll