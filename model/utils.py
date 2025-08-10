from config import *
import numpy as np
import pretty_midi
from config import * # Import all your configuration variables
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def drum_to_pianoroll(instrument):
    """
    Create a pianoroll for a drum track manually because for instrument.is_drum 
    the function: instrument.get_pianoroll() doesn't work.
    """
    end_time = max(note.end for note in instrument.notes)
    n_frames = int(end_time * FS) + 1
    pianoroll = np.zeros((128, n_frames)) 

    for note in instrument.notes:
        # if not (MIN_MIDI_NOTE <= note.pitch <= MAX_MIDI_NOTE):
        #     continue
        start = int(note.start * FS)
        end = int(note.end * FS)

        pitch = note.pitch  
        velocity = note.velocity

        # Fill values in the piano roll
        pianoroll[pitch, start:end] = velocity
    
    # Crop tensor to MIN_MIDI_NOTE and MAX_MIDI_NOTE
    pianoroll = pianoroll[MIN_MIDI_NOTE:MAX_MIDI_NOTE + 1, :]
    
    # Flip the pianoroll so it looks like a piano roll
    # with the lowest pitch at the bottom
    flipped_pianoroll = np.flip(pianoroll, axis=0)

    return flipped_pianoroll


def pianoroll_to_instrument(piano_roll, fs, program=0):
    """
    Converts a piano roll numpy array into a PrettyMIDI Instrument object.
    (This version is robust against silent piano rolls).
    """
    instrument = pretty_midi.Instrument(program=program, is_drum=True, name="Generated Drums")
    
    notes, frames = piano_roll.shape
    piano_roll = np.pad(piano_roll, [(0, 0), (0, 1)], 'constant')
    velocity_threshold = 10 

    for pitch in range(notes):
        # Find the frames where this pitch is active
        frames_where_pitch_is_on = np.where(piano_roll[pitch] > velocity_threshold)[0]
        
        if len(frames_where_pitch_is_on) == 0:
            continue
        
        frame_diffs = np.diff(frames_where_pitch_is_on)
        
        # Find the start of each note event
        start_frames = np.where(frame_diffs > 1)[0]
        start_frames = np.append(0, start_frames + 1)
        start_frames = frames_where_pitch_is_on[start_frames]
        
        # Find the end of each note event
        end_frames = np.where(frame_diffs > 1)[0]
        end_frames = frames_where_pitch_is_on[end_frames]
        end_frames = np.append(end_frames, frames_where_pitch_is_on[-1])

        for i in range(len(start_frames)):
            start_time = start_frames[i] / fs
            end_time = (end_frames[i] + 1) / fs
            
            velocity = int(piano_roll[pitch, start_frames[i]])

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

    return instrument


def pianoroll_tensor_to_midi(pianoroll_tensor, output_path):
    """
    Converts a piano roll tensor back into a MIDI file.
    (This function now uses our new helper).
    """
    if pianoroll_tensor.is_cuda:
        pianoroll_tensor = pianoroll_tensor.cpu()
        
    if pianoroll_tensor.dim() == 4:
        pianoroll_tensor = pianoroll_tensor.squeeze(0)
    
    pianoroll_tensor = pianoroll_tensor.squeeze(0)

    pianoroll_velocities = pianoroll_tensor * 127.0
    pianoroll_np = pianoroll_velocities.numpy().astype(np.int16)

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)
    
    drum_instrument = pianoroll_to_instrument(pianoroll_np, fs=FS, program=0)
    
    midi_data.instruments.append(drum_instrument)
    midi_data.write(output_path)
    print(f"Successfully saved MIDI file to {output_path}")

def visualize_latent_space(model, dataloader, output_filename="latent_space.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print("\n--- Visualising latent space ---")
    all_z_means = []
    all_avg_pitches = [] 

    with torch.no_grad():
        for pianorolls, lengths in dataloader:
            pianorolls = pianorolls.to(device)

            mean, _ = model.encoder(pianorolls, lengths)
            
            all_z_means.append(mean.cpu().numpy())

            for i in range(pianorolls.size(0)):
                single_pianoroll = pianorolls[i, :, :lengths[i]].cpu().numpy()
                note_indices = np.where(single_pianoroll > 0.1)
                if len(note_indices[0]) > 0:
                    avg_pitch = np.mean(note_indices[0])
                    all_avg_pitches.append(avg_pitch)
                else:
                    all_avg_pitches.append(0)

    all_z_means = np.concatenate(all_z_means, axis=0)
    all_avg_pitches = np.array(all_avg_pitches)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_z_means)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        z_2d[:, 0], 
        z_2d[:, 1], 
        c=all_avg_pitches, 
        cmap='viridis',
        alpha=0.7,
        s=15
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Average velocity')

    ax.set_title('Latent Space visualisation', fontsize=16)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.grid(True)

    plt.savefig(output_filename, dpi=300)
    print(f"Saving plot to: {output_filename}\n")
    plt.show()


