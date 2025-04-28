import mido

# Load a MIDI file
mid = mido.MidiFile('your_file.mid')

# Print basic info
print(f"Type: {mid.type}, Ticks per beat: {mid.ticks_per_beat}, Tracks: {len(mid.tracks)}")

# Iterate over messages
for i, track in enumerate(mid.tracks):
    print(f'\nTrack {i}: {track.name}')
    for msg in track:
        print(msg)

