from pathlib import Path
from basic_pitch.inference import predict_and_save
import os
from basic_pitch import ICASSP_2022_MODEL_PATH


def convert_audio_to_midi(wav_files, song_name):

    output_dir = Path(f"./midi/{song_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_and_save(audio_path_list=wav_files,
                     output_directory=f"./midi/{song_name}",
                     save_midi=True,
                     sonify_midi=False,
                     save_model_outputs=False,
                     save_notes=False,
                    #  melodia_trick=True,
                     model_or_model_path=ICASSP_2022_MODEL_PATH
                     )


separated_wav_dir = "./separated/htdemucs_ft/"
all_separated_songs = os.listdir(separated_wav_dir)

print(f"{all_separated_songs = }")

for song in all_separated_songs:
    song_dir = Path(os.path.join(separated_wav_dir,song))
    separated_wav_files = list(song_dir.glob("*.wav"))

    convert_audio_to_midi(separated_wav_files, song)

