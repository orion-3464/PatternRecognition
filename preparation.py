import librosa
from pathlib import Path

# Step 2
def data_parser(dir = 'pr_lab2_data/digits'):
    p = Path(dir)

    rosa_wavs = []
    speakers = []
    digits = []

    files = list(p.glob('*.wav')) # PosixPath('pr_lab2_data/digits/seven11.wav')

    for file in files:
        data, sr = librosa.load(file)
        rosa_wavs.append(data)

        filename = file.parts[-1] # seven11.wav
        speakers.append(''.join([d for d in filename if d.isdigit()]))

        digits.append(''.join([char for char in filename if char.isalpha()]))

    return rosa_wavs, speakers, digits


if __name__ == '__main__':
    wavs, speakers, digits = data_parser()

