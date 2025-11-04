import os
from glob import glob
import librosa
import pomegranate

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("/")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers

def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = {
        "0" : 0, 
        "1" : 0,
        "2" : 0, 
        "3" : 0, 
        "4" : 0, 
        "5" : 0, 
        "6" : 0, 
        "7" : 0, 
        "8" : 0, 
        "9" : 0
    }

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices.keys():
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
#            test_indices[idx] = test_indices[idx] + 1
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

#    print(f"Train dataset's size is {len(X_train)}") # <- DEBUGGING
#    for key, values in test_indices.items():
#        print(f"Train dataset has {values} samples of digit {key}")
#        print(f"Train dataset has {values} samples of digit {key}")
    return X_train, X_test, y_train, y_test, spk_train, spk_test

if __name__ == '__main__':
    # Step 9
    wavs, Fs, ids, y, speakers = parse_free_digits('recordings')

    # 60 digits of each one in train, 2400 samples - 80% (3000 in total)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(wavs, ids, speakers, y) 

    # Step 10

    
