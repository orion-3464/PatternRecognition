import os
from glob import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import numpy as np
from tqdm import tqdm

# ========== parser.py ==========

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


def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            y=wav, sr=Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T
        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames


def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"]

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test


def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))

    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled

    return scale


def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

# ========== hmm.py ==========

# TODO: YOUR CODE HERE
# Play with diffrent variations of parameters in your experiments
n_states = 2  # the number of HMM states
n_mixtures = 2  # the number of Gaussians
gmm = True  # whether to use GMM or plain Gaussian
covariance_type = "diag"  # Use diagonal covariange


# Gather data separately for each digit
def gather_in_dic(X, labels, spk):
    dic = {}
    for dig in set(labels):
        x = [X[i] for i in range(len(labels)) if labels[i] == dig]
        lengths = [len(i) for i in x]
        y = [dig for _ in range(len(x))]
        s = [spk[i] for i in range(len(labels)) if labels[i] == dig]
        dic[dig] = (x, lengths, y, s)
    return dic

# Help code's extract features
def extract_features(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            y=wav, sr=Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T
        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames

# Help code's parser function in parser.py
def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test

def create_data():
    X, X_test, y, y_test, spk, spk_test = parser("recordings", n_mfcc=13)

    # TODO: YOUR CODE HERE
    (
        X_train,
        X_val,
        y_train,
        y_val,
        spk_train,
        spk_val,
    ) =  train_test_split(X, y, spk, test_size=0.2, random_state=42, stratify=y) # split X into a 80/20 train validation split
    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))

    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels


def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):
    # TODO: YOUR CODE HERE
    dists = []
    for _ in range(n_states):
        distributions = ...  # n_mixtures gaussian distributions
        a = GeneralMixtureModel(distributions, verbose=True).fit(
            np.concatenate(X)
        )  # Concatenate all frames from all samples into a large matrix
        dists.append(a)
    return dists


def initialize_and_fit_normal_distributions(X, n_states):
    dists = []
    for _ in range(n_states):
        # TODO: YOUR CODE HERE
        d = ...  # Fit a normal distribution on X
        dists.append(d)
    return dists


def initialize_transition_matrix(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    return ...


def initialize_starting_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    return ...


def initialize_end_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    return ...


def train_single_hmm(X, emission_model, digit, n_states):
    A = initialize_transition_matrix(n_states)
    start_probs = initialize_starting_probabilities(n_states)
    end_probs = initialize_end_probabilities(n_states)
    data = [x.astype(np.float32) for x in X]

    model = DenseHMM(
        distributions=emission_model,
        edges=A,
        starts=start_probs,
        ends=end_probs,
        verbose=True,
    ).fit(data)
    return model


def train_hmms(train_dic, labels):
    hmms = {}  # create one hmm for each digit

    for dig in labels:
        X, _, _, _ = train_dic[dig]
        # TODO: YOUR CODE HERE
        emission_model = ...
        hmms[dig] = ...
    return hmms


def evaluate(hmms, dic, labels):
    pred, true = [], []
    for dig in labels:
        X, _, _, _ = dic[dig]
        for sample in X:
            ev = [0] * len(labels)
            sample = np.expand_dims(sample, 0)
            for digit, hmm in hmms.items():
                # TODO: YOUR CODE HERE
                logp = ...  # use the hmm.log_probability function
                ev[digit] = logp

            # TODO: YOUR CODE HERE
            predicted_digit = ...  # Calculate the most probable digit
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true

# Help code plot confusion matrix function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



if __name__ == '__main__':
    # Step 9
    train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()

    # Step 10

    
