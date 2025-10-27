import librosa
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy

Fs = 16000
n1 = 'two' # Kordas Nikolaos (03121032)
n2 = 'five' # Kritharidis Konstantinos (03121045)

plt.rcParams['text.usetex'] = True

def data_parser(dir = 'pr_lab2_data/digits'):
    p = Path(dir)

    rosa_wavs = []
    speakers = []
    digits = []

    files = list(p.glob('*.wav')) # e.g. PosixPath('pr_lab2_data/digits/seven11.wav')

    for file in files:
        data, _ = librosa.load(file, sr=None)
        rosa_wavs.append(data)

        filename = file.parts[-1] # e.g. seven11.wav
        speakers.append(''.join([d for d in filename if d.isdigit()]))

        digits.append(''.join([char for char in filename.split('.')[-2] if char.isalpha()]))

    return rosa_wavs, speakers, digits

def extract_features(wavs):
    mfccs = []
    win_len = 25 * Fs // 1000 
    step = 10 * Fs // 1000
    n_features = 13
    deltas = []
    delta_deltas = []

    for wav in wavs:
        mfcc = librosa.feature.mfcc(y=wav, sr=Fs, n_mfcc=n_features, hop_length=step, win_length=win_len)
        mfccs.append(mfcc)
        deltas.append(librosa.feature.delta(mfcc, order=1))
        delta_deltas.append(librosa.feature.delta(mfcc, order=2))

    return mfccs, deltas, delta_deltas

def plot_histgrams(wavs, digits, mfccs):
    mfccs_n1_0, mfccs_n1_1, mfccs_n2_0, mfccs_n2_1 = [], [], [], []

    for i in range(len(wavs)):
        if digits[i] == n1: 
            mfccs_n1_0.append(mfccs[i][0])
            mfccs_n1_1.append(mfccs[i][1])
        elif digits[i] == n2: 
            mfccs_n2_0.append(mfccs[i][0])
            mfccs_n2_1.append(mfccs[i][1])
    
    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize=(10, 10))

    axs[0][0].hist(mfccs_n1_0)
    axs[0][1].hist(mfccs_n1_1)
    axs[1][0].hist(mfccs_n2_0)
    axs[1][1].hist(mfccs_n2_1)

    axs[0][0].grid(visible=True)
    axs[0][0].set_xlabel(r'1\textsuperscript{st} MFCC value')
    axs[0][0].set_ylabel(r'Frequencies')
    axs[0][0].set_title(r'1\textsuperscript{st} MFCC for digit n\textsubscript{1}=2')

    axs[0][1].grid(visible=True)
    axs[0][1].set_xlabel(r'2\textsuperscript{nd} MFCC value')
    axs[0][1].set_ylabel(r'Frequencies')
    axs[0][1].set_title(r'2\textsuperscript{nd} MFCC for digit n\textsubscript{1}=2')

    axs[1][0].grid(visible=True)
    axs[1][0].set_xlabel(r'1\textsuperscript{st} MFCC value')
    axs[1][0].set_ylabel(r'Frequencies')
    axs[1][0].set_title(r'1\textsuperscript{st} MFCC for digit n\textsubscript{2}=5')

    axs[1][1].grid(visible=True)
    axs[1][1].set_xlabel(r'2\textsuperscript{nd} MFCC value')
    axs[1][1].set_ylabel(r'Frequencies')
    axs[1][1].set_title(r'2\textsuperscript{nd} MFCC for digit n\textsubscript{2}=5')

    fig.suptitle(r'\textbf{1\textsuperscript{st} and 2\textsuperscript{nd} MFCCs of digits n\textsubscript{1}=2 and n\textsubscript{2}=5}', fontsize=16)
    plt.savefig("plots/histograms_step4.png")
    plt.show()

def extract_mfscs1(mfccs): # MFCC to Mel
    mfscs = []

    for mfcc in mfccs:
        mfscs.append(librosa.feature.inverse.mfcc_to_mel(mfcc))
    return mfscs

def extract_mfscs2(mfccs): # Inverse DCT
    mfscs = []

    for mfcc in mfccs:
        mfscs.append(librosa.db_to_power(scipy.fft.idct(mfcc, type=2, norm="ortho", axis=0)))
    return mfscs

def extract_mfscs(wavs): # Forward
    mfscs = []
    win_len = 25 * Fs // 1000 
    step = 10 * Fs // 1000
    n_features = 13

    for wav in wavs:
        mfscs.append(librosa.feature.melspectrogram(y=wav, sr=Fs, win_length=win_len, hop_length=step, n_mels=n_features))
    return mfscs

if __name__ == '__main__':
    # Step 2
    wavs, speakers, digits = data_parser()

    # Step 3
    mfccs, deltas, delta_deltas = extract_features(wavs)
    
    # Step 4.1
    plot_histgrams(wavs, digits, mfccs)

    # Step 4.2
    mfscs = extract_mfscs(mfccs)

    mfscs1 = extract_mfscs1(wavs)
    mfscs2 = extract_mfscs2(wavs)

    print(len(mfscs[0][0]), mfscs[0][0])
    print(len(mfscs1[0][0]), mfscs1[0][0])
    print(len(mfscs2[0][0]), mfscs2[0][0])
    