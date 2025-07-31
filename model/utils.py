import pretty_midi
from config import *

def print_model_details(model):
    """    
    Prints the details of the model including its architecture and number of trainable parameters.
    """
    print(f"Model details for {model.__class__.__name__}:\n")
    print(f"{model}\n")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")
    print(80*"_", "\n")        

def tensor_to_midi(tensor, save_to_midi, save_path, velocity_threshold=0.1):
    midi_file = pretty_midi.PrettyMIDI()
    
    instrument_map = [
        {'program': 0,  'is_drum': True,  'name': 'Drums'},
        {'program': 0,  'is_drum': False, 'name': 'Piano'},
        {'program': 33, 'is_drum': False, 'name': 'Bass'},
        {'program': 25, 'is_drum': False, 'name': 'Guitar'},
        {'program': 48, 'is_drum': False, 'name': 'Others'}
    ]

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # (1, I, P, T) → (I, P, T)

    for i in range(tensor.shape[0]):
        pr = tensor[i].cpu().numpy().T  # (T, P)
        info = instrument_map[i]
        instrument = pretty_midi.Instrument(program=info['program'], is_drum=info['is_drum'], name=info['name'])

        for pitch in range(pr.shape[1]):
            note_on_time = None
            peak_velocity = 0.0
            for t in range(pr.shape[0]):
                vel = pr[t, pitch]
                is_on = vel > velocity_threshold

                if is_on and note_on_time is None:
                    note_on_time = t / FS
                    peak_velocity = vel
                elif is_on:
                    peak_velocity = max(peak_velocity, vel)
                elif not is_on and note_on_time is not None:
                    end = t / FS
                    velocity = int(peak_velocity * 127)
                    if velocity > 0:
                        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch + MIN_MIDI_NOTE, start=note_on_time, end=end))
                    note_on_time = None
                    peak_velocity = 0.0

            if note_on_time is not None:
                end = pr.shape[0] / FS
                velocity = int(peak_velocity * 127)
                if velocity > 0:
                    instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch + MIN_MIDI_NOTE, start=note_on_time, end=end))

        midi_file.instruments.append(instrument)

    if save_to_midi:
        midi_file.write(save_path)
        print(f"✅ MIDI saved to: {save_path}")

    return midi_file

