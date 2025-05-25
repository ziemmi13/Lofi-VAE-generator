import subprocess
from pathlib import Path

def download_wav(yt_playlist_url):
    # Define output directory and ensure it exists
    output_dir = Path("./wav_files")

    # yt-dlp output template — audio files go to output_dir
    output_template = str(output_dir / "%(title)s.%(ext)s")

    yt_dlp_command = [
        "yt-dlp.exe",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "10",
        "--split-chapters",
        "--no-keep-video",
        "-o", output_template,
        yt_playlist_url
    ]

    print("Downloading and extracting audio...")
    subprocess.run(yt_dlp_command, check=True)
    print(f"Download complete. Files saved to: {output_dir}")

    all_wav_files = list(output_dir.glob("*.wav"))

    return all_wav_files


def separate_stems(wav_files):

    for wav_file in wav_files:

        print(f"Separating stems from: {wav_file.name}")
        demucs_command = [
            "demucs",
            "-n", "htdemucs_ft",
            str(wav_file),
            "--shifts", "5"
        ]
        subprocess.run(demucs_command, check=True)
        print(f"Finished separating: {wav_file.name}")

# url = "https://www.youtube.com/watch?v=54eT20uujiM&list=PLn3XLGKpvI5uhNq5PWshtC6LQZWSw4OvC"
url = "https://www.youtube.com/watch?v=O3nB-ld08JI"

wav_files = download_wav(yt_playlist_url=url)
separate_stems(wav_files=wav_files)