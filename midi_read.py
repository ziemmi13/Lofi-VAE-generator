import pretty_midi
import numpy as np

import music21


# Wczytaj plik MIDI
midi_data = pretty_midi.PrettyMIDI(r'C:\Users\Hyperbook\Desktop\STUDIA\SEM III\Projekt zespolowy\utils\midi\short_midi.mid')

# Zdobądź wszystkie nuty (w czasie)
notes = []
for instrument in midi_data.instruments:
    for note in instrument.notes:
        notes.append([note.start, note.end, note.pitch, note.velocity])

# Możesz zamienić te dane na odpowiednią reprezentację
# notes = np.array(notes)

# Na przykład, możesz wyciągnąć spektrogram
spectrogram = midi_data.get_piano_roll()

# Teraz masz reprezentację, którą możesz użyć w sieci neuronowej

with open('notes.txt', 'a') as f:
    f.write(str(notes))


# my_score = music21.converter.parse(midi_data)
# k = my_score.analyze('key')
# print(k.name)  # Prints the key name (e.g., 'C Major', 'G minor')

