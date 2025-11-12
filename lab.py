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
from torch.utils.data import Dataset
import torch.nn as nn
import math
import torch
from torch.utils.data import DataLoader

np.random.seed(42)

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
    return X_train, X_test, y_train, y_test, spk_train, spk_test

# ========== hmm.py ==========

# TODO: YOUR CODE HERE
# Play with diffrent variations of parameters in your experiments
n_states = 4  # the number of HMM states
n_mixtures = 5  # the number of Gaussians
n_iters = 100 # max number of iterations for EM
tol = 0.5
gmm = True  # whether to use GMM or plain Gaussian
covariance_type = "diag"  # Use diagonal covariance


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

    # Scaling data based on the training set
    scale_fn = make_scale_fn(X_train)
    X_train = scale_fn(X_train)
    X_val = scale_fn(X_val)
    X_test = scale_fn(X_test)

    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))

    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels


def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):
    # TODO: YOUR CODE HERE
    dists = []
    all_data = np.concatenate(X, dtype=np.float32) # Concatenate all frames from all samples into a large matrix, shape (N, D)
    D = all_data.shape[1]

    for _ in range(n_states):
        distributions = [Normal(means=np.random.randn(D).astype(np.float32), covs=np.ones(D, dtype=np.float32), covariance_type=covariance_type) for _ in range(n_mixtures)] # n_mixtures gaussian distributions
        a = GeneralMixtureModel(distributions, verbose=True, max_iter=n_iters, random_state=42, tol=tol).fit(all_data) # default log propability threshold is 0.1        
        dists.append(a)
    return dists


def initialize_and_fit_normal_distributions(X, n_states):
    dists = []
    all_data = np.concatenate(X, dtype=np.float32) # Concatenate all frames from all samples into a large matrix, shape (N, D)
    D = all_data.shape[1]

    for _ in range(n_states):
        # TODO: YOUR CODE HERE
        d = Normal(means=np.random.randn(D), covs=np.ones(D), covariance_type=covariance_type).fit(all_data)  # Fit a normal distribution on X
        dists.append(d)
    return dists


def initialize_transition_matrix(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    A = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n_states):
        if i+1 < n_states:
            A[i][i] = np.float32(0.8)
            A[i][i+1] = np.float32(0.2)
        else:
            A[i][i] = np.float32(1.0)
    return A


def initialize_starting_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    pi = np.zeros(n_states, dtype=np.float32)
    pi[0] = np.float32(1.0)
    return pi


def initialize_end_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    pi_end = np.zeros(n_states, dtype=np.float32)
    pi_end[-1] = np.float32(1.0)
    return pi_end


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
        max_iter=n_iters,
        tol=tol,
        random_state=42
    ).fit(data)
    return model


def train_hmms(train_dic, labels):
    hmms = {}  # create one hmm for each digit

    for dig in labels:
        X, _, _, _ = train_dic[dig]
        # TODO: YOUR CODE HERE
        emission_model = None
        if gmm: emission_model = initialize_and_fit_gmm_distributions(X, n_states=n_states, n_mixtures=n_mixtures)
        else: emission_model = initialize_and_fit_normal_distributions(X, n_states=n_states)
        hmms[dig] = train_single_hmm(X, emission_model, dig, n_states)
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
                logp = hmm.log_probability(sample)  # use the hmm.log_probability function
                ev[digit] = logp

            # TODO: YOUR CODE HERE
            predicted_digit = np.argmax(ev)  # Calculate the most probable digit
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true

# ========== plot_confusion_matrix.py ==========
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

def hmm_tuning(max_n_states, max_n_mixtures):
    best_n_states = None
    best_n_mixtures = None
    best_accuracy = 0
    best_hmms = None
    best_pred_val = None
    best_true_val = None
    for n_states in range(1, max_n_states+1):
        for n_mixtures in range(1, max_n_mixtures+1):
            np.random.seed(42)
            hmms = train_hmms(train_dic, labels)
            pred_val, true_val = evaluate(hmms, val_dic, labels)
            val_accuracy = accuracy_score(true_val, pred_val)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_n_states = n_states
                best_n_mixtures = n_mixtures
                best_hmms = hmms
                best_pred_val = pred_val
                best_true_val = true_val

    n_states = best_n_states
    n_mixtures = best_n_mixtures

    print(f"Best: states = {n_states}, mixtures = {n_mixtures}.")

    return best_hmms, best_accuracy, best_pred_val, best_true_val

# ========== lstm.py ==========

output_dim = 10  # number of digits
# TODO: YOUR CODE HERE
# Play with variations of these hyper-parameters and report results
rnn_size = 64
num_layers = 2
bidirectional = True
dropout = 0.4
batch_size = 32
patience = 3
epochs = 15
lr = 1e-3
weight_decay = 0.0


class EarlyStopping(object):
    def __init__(self, patience, mode="min", base=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode

    def stop(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Decrease patience if the metric hs not improved
        # Stop when patience reaches zero
        return False

    def has_improved(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Check if the metric has improved
        return False


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
        feats: Python list of numpy arrays that contain the sequence features.
               Each element of this list is a numpy array of shape seq_length x feature_dimension
        labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        # TODO: YOUR CODE HERE
        self.lengths = [len(i) for i in feats]

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype("int64")
        

    def zero_pad_and_stack(self, x: np.ndarray) -> np.ndarray:
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array
        returns
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = np.zeros(len(x), max(self.lengths), x[0].shape[1])
        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #
        for i, seq in enumerate(x):
            padded[i, :len(seq), :] = seq

        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=False,
        dropout=0.0,
    ):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers


    def forward(self, x, lengths):
        """
        x : 3D numpy array of dimension N x L x D
            N: batch index
            L: sequence index
            D: feature index

        lengths: N x 1
        """

        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #

        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        last_outputs: torch.Tensor = ...
        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
        Returns the last output of the LSTM taking into account the zero padding
        """
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        # Index of the last output for each sequence.
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(outputs.size(0), outputs.size(2))
            .unsqueeze(1)
        )
        return outputs.gather(1, idx).squeeze()


def create_dataloaders(batch_size):
    X, X_test, y, y_test, spk, spk_test = parser("recordings", n_mfcc=13)

    X_train, X_val, y_train, y_val, spk_train, spk_val = train_test_split(
        X, y, spk, test_size=0.2, stratify=y
    )

    trainset = FrameLevelDataset(X_train, y_train)
    validset = FrameLevelDataset(X_val, y_val)
    testset = FrameLevelDataset(X_test, y_test)
    # TODO: YOUR CODE HERE
    # Initialize the training, val and test dataloaders (torch.utils.data.DataLoader)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) # shuffle true only in train
    val_dataloader = DataLoader(validset, batch_size=batch_size)
    test_dataloader = DataLoader(testset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def training_loop(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for num_batch, batch in enumerate(train_dataloader):
        features, labels, lengths = batch
        # TODO: YOUR CODE HERE
        # zero grads in the optimizer
        # run forward pass
        # calculate loss
        loss = ...
        # TODO: YOUR CODE HERE
        # Run backward pass
        running_loss += loss.item()
        num_batches += 1
    train_loss = running_loss / num_batches
    return train_loss


def evaluation_loop(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    y_pred = torch.empty(0, dtype=torch.int64)
    y_true = torch.empty(0, dtype=torch.int64)
    with torch.no_grad():
        for num_batch, batch in enumerate(dataloader):
            features, labels, lengths = batch

            # TODO: YOUR CODE HERE
            # Run forward pass
            logits = ...
            # calculate loss
            loss = ...
            running_loss += loss.item()
            # Predict
            outputs = ...  # Calculate the argmax of logits
            y_pred = torch.cat((y_pred, outputs))
            y_true = torch.cat((y_true, labels))
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true


def train(train_dataloader, val_dataloader, criterion):
    # TODO: YOUR CODE HERE
    input_dim = train_dataloader.dataset.feats.shape[-1]
    model = BasicLSTM(
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    # TODO: YOUR CODE HERE
    # Initialize AdamW
    optimizer = ...

    early_stopping = EarlyStopping(patience, mode="min")
    for epoch in range(epochs):
        training_loss = training_loop(model, train_dataloader, optimizer, criterion)
        valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)

        # TODO: Calculate and print accuracy score
        valid_accuracy = ...
        print(
            "Epoch {}: train loss = {}, valid loss = {}, valid acc = {}".format(
                epoch, training_loss, valid_loss, valid_accuracy
            )
        )
        if early_stopping.stop(valid_loss):
            print("early stopping...")
            break

    return model

if __name__ == '__main__':
    # Step 9
    train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()

    # Step 10 - 11
    hmms = train_hmms(train_dic, labels)
    labels = list(set(y_train))

    # Step 12
    hmms, val_accuracy, pred_val, true_val = hmm_tuning(4, 5)
    
    print()
    print(f"Accuracy score on the validation set is: {val_accuracy}")

    pred_test, true_test = evaluate(hmms, test_dic, labels)
    test_accuracy = accuracy_score(true_test, pred_test)
    print()
    print(f"Accuracy score on the test set is: {test_accuracy}")
    
    # Step 13
    cm_val = confusion_matrix(true_val, pred_val) 
    plot_confusion_matrix(cm_val, classes=labels)
    plt.show()

    cm_test = confusion_matrix(true_test, pred_test) 
    plot_confusion_matrix(cm_test, classes=labels)
    plt.show()
    
    # Step 14
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)

    criterion = nn.BCELoss()
    model = train(train_dataloader, val_dataloader, criterion)