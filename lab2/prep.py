import os
import pandas as pd
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Step 0
    os.listdir("../input/patreco3-multitask-affective-music/data/")

    # Step 1
    train_labels = pd.read_csv("/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt", sep="\t")
    train_labels.groupby("Genre").sample(n=1, random_state=42)

    spec1 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/11868.fused.full.npy')
    spec2 = np.load('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/18416.fused.full.npy')
    # Printing spec1 and spec2 shapes according to help code
    print(spec1.shape) 
    print(spec2.shape)

    mel1, chroma1 = spec1[:128], spec1[128:]
    mel2, chroma2 = spec2[:128], spec2[128:]
    print(mel1.shape, chroma1.shape)
    print(mel2.shape, chroma2.shape)

    fig, axs = plt.subplots(2)
    img1 = librosa.display.specshow(mel1, x_axis='time', y_axis='linear', ax=axs[0])
    img2 = librosa.display.specshow(mel2, x_axis='time', y_axis='linear', ax=axs[1])
    axs[0].set(title='Hip-Hop Spectrogram')
    axs[1].set(title='Punk Spectrogram')
    plt.tight_layout()
    fig.colorbar(img1, ax=axs, format="%+2.f dB")

