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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import math
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

np.random.seed(42)
torch.manual_seed(42)

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
n_iters = 250 # max number of iterations for EM
tol = 0.1
gmm = True  # whether to use GMM or plain Gaussian
covariance_type = "diag"  # Use diagonal covariance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    all_data = np.concatenate(X, dtype=np.float32)
    D = all_data.shape[1]

    global_mean = np.mean(all_data, axis=0)
    global_std = np.std(all_data, axis=0)
    
    variance_floor = 1e-6
    base_variance = np.maximum(global_std**2, variance_floor).astype(np.float32)

    for _ in range(n_states):
        distributions = []
        for _ in range(n_mixtures):
            means = global_mean + 0.1 * np.random.randn(D).astype(np.float32)
            
            covs = base_variance.copy()
            
            distributions.append(
                Normal(means=means, covs=covs, covariance_type=covariance_type)
            )
        
        gmm_model = GeneralMixtureModel(
            distributions, 
            verbose=False, 
            max_iter=n_iters, 
            random_state=42, 
            tol=tol
        ).fit(all_data)
        dists.append(gmm_model)

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
    A = np.full((n_states, n_states), 1e-6 ,dtype=np.float32)
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


def train_hmms(train_dic, labels, n_states, n_mixtures):
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
    plt.show()

def hmm_tuning(max_n_states, max_n_mixtures):
    # Storage for all model metrics
    metrics = {
        'n_states': [],
        'n_mixtures': [],
        'accuracy': [],
        'precision': [],
        'confusion_matrices': [],
        'predictions': [],
        'true_labels': []
    }
    
    best_accuracy = 0
    best_metrics = {}
    best_model = None

    for n_states in range(1, max_n_states + 1):
        for n_mixtures in range(1, max_n_mixtures + 1):
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Train and evaluate model
            hmms = train_hmms(train_dic, labels, n_states, n_mixtures)  # Assumes these params are used
            pred_val, true_val = evaluate(hmms, val_dic, labels)
            
            # Calculate metrics
            val_accuracy = accuracy_score(true_val, pred_val)
            val_precision = precision_score(true_val, pred_val, average='weighted', zero_division=0)
            cm = confusion_matrix(true_val, pred_val)

            # Store metrics
            metrics['n_states'].append(n_states)
            metrics['n_mixtures'].append(n_mixtures)
            metrics['accuracy'].append(val_accuracy)
            metrics['precision'].append(val_precision)
            metrics['confusion_matrices'].append(cm)
            metrics['predictions'].append(pred_val)
            metrics['true_labels'].append(true_val)

            # Update best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_metrics = {
                    'n_states': n_states,
                    'n_mixtures': n_mixtures,
                    'accuracy': val_accuracy,
                    'precision': val_precision,
                    'confusion_matrix': cm,
                    'predictions': pred_val,
                    'true_labels': true_val,
                    'model': hmms
                }
                best_model = hmms

    print(f"Best: states = {best_metrics['n_states']}, mixtures = {best_metrics['n_mixtures']}, "
          f"accuracy = {best_metrics['accuracy']:.4f}")

    return best_metrics, metrics, best_model

def plot_tuning_results(metrics, best_metrics):
    n_configs = len(metrics['accuracy'])
    
    max_states = max(metrics['n_states'])
    max_mixtures = max(metrics['n_mixtures'])
    
    accuracy_matrix = np.zeros((max_states, max_mixtures))
    for i in range(n_configs):
        states = metrics['n_states'][i] - 1  
        mixtures = metrics['n_mixtures'][i] - 1 
        accuracy_matrix[states, mixtures] = metrics['accuracy'][i]
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(accuracy_matrix, cmap='YlGnBu', aspect='auto')
    plt.colorbar(im, label='Accuracy')
    plt.title('Validation Accuracy by Hyperparameters')
    plt.xlabel('Number of Mixtures')
    plt.ylabel('Number of States')
    
    plt.xticks(np.arange(max_mixtures), np.arange(1, max_mixtures + 1))
    plt.yticks(np.arange(max_states), np.arange(1, max_states + 1))
    
    for i in range(max_states):
        for j in range(max_mixtures):
            text = plt.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black")
    plt.show()

    plt.figure(figsize=(12, 6))
    config_indices = np.arange(n_configs)
    
    plt.plot(config_indices, metrics['accuracy'], 'o-', label='Accuracy', linewidth=2, markersize=6)
    plt.plot(config_indices, metrics['precision'], 's-', label='Precision', linewidth=2, markersize=6)
    
    plt.xlabel('Configuration Index')
    plt.ylabel('Score')
    plt.title('Accuracy and Precision Across All Configurations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(config_indices)
    plt.show()

    plt.figure(figsize=(8, 6))
    cm = best_metrics['confusion_matrix']
    im = plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = plt.text(j, i, str(cm[i, j]),
                           ha="center", va="center", color="black")
    
    plt.title(f'Best Model Confusion Matrix\n'
              f"States: {best_metrics['n_states']}, Mixtures: {best_metrics['n_mixtures']}\n"
              f"Accuracy: {best_metrics['accuracy']:.4f}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# ========== lstm.py ==========

output_dim = 10  # number of digits
# TODO: YOUR CODE HERE
# Play with variations of these hyper-parameters and report results
rnn_size = 64
num_layers = 2
dropout = 0.4
batch_size = 32
patience = 10
epochs = 750
lr = 2e-5
weight_decay = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_early_stopping = False
use_packing=False

checkpoint_filename = "best_model.pt"

class EarlyStopping(object):
    def __init__(self, patience, mode="min", base=None, base_model=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode
        self.pickle = None if base_model == None else torch.save(base_model.state_dict(), checkpoint_filename)

    def stop(self, value: float, model) -> bool:
        # TODO: YOUR CODE HERE
        # Decrease patience if the metric hs not improved
        if self.has_improved(value):
            self.best = value
            self.patience_left = self.patience  # reset patience
            self.model = model
            self.pickle = torch.save(model.state_dict(), checkpoint_filename)
        else:
            self.patience_left -= int(use_early_stopping)
        
        # Stop when patience reaches zero
        return self.patience_left <= 0

    def has_improved(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Check if the metric has improved
        if self.best is None:
            return True

        if self.mode == "min":
            return value < self.best
        elif self.mode == "max":
            return value > self.best
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Use 'min' or 'max'.")


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
        #padded = np.zeros(len(x), max(self.lengths), x[0].shape[1])
        padded = np.zeros((len(x), max(self.lengths), x[0].shape[1]))
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
        dropout=dropout,
    ):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.feature_size, output_dim)


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
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, dtype=torch.int64)
        
        if use_packing:
            # sort the features in decreasing length for packing
            lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
            x_sorted = x[sort_idx]
			# pack sequence to avoid processing the padding
            packed_input = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
            # get packed output
            packed_output, _ = self.lstm(packed_input)
			# retrieve sequence in its original (padded) format
            outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
            # retrieve the output in its original order
            _, unsort_idx = torch.sort(sort_idx)
            outputs = outputs[unsort_idx]
        else:
            outputs, _ = self.lstm(x)

        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        last_outputs = self.last_timestep(outputs, lengths, self.bidirectional)

        last_outputs = self.dropout(last_outputs)
        last_outputs = self.output_layer(last_outputs)

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
        features = features.to(device).float()
        labels = labels.to(device)
        lengths = lengths.to(device)

        # zero grads in the optimizer
        optimizer.zero_grad()
        # run forward pass
        outputs = model(features, lengths)
        # calculate loss
        loss = criterion(outputs, labels)

        # TODO: YOUR CODE HERE
        # Run backward pass
        loss.backward()
        optimizer.step()

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
            features = features.to(device).float()
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Run forward pass
            logits = model(features, lengths)
            # calculate loss
            loss = criterion(logits, labels)
            running_loss += loss.item()
            # Predict
            outputs = torch.argmax(logits, dim=1) # Calculate the argmax of logits
            y_pred = torch.cat((y_pred, outputs.cpu()))
            y_true = torch.cat((y_true, labels.cpu()))
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true


def train(models, optimizers, train_dataloader, val_dataloader, criterion):
    # TODO: YOUR CODE HERE
    metrics = {
        "training_loss": {},
        "validation_loss": {},
        "validation_accuracy": {},
    }

    confusion_matrices = {}
    
    best_model_name, best_val_acc = None, -1

    for model_name, model, optimizer in zip(models.keys(), models.values(), optimizers.values()):
        model.to(device)

        early_stopping = EarlyStopping(patience, mode="min", base_model=None)

        print(f"Training in {device}")

        metrics["training_loss"][model_name] = []
        metrics["validation_loss"][model_name] = []
        metrics["validation_accuracy"][model_name] = []
        for epoch in range(epochs):
            training_loss = training_loop(model, train_dataloader, optimizer, criterion)
            valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)

            # TODO: Calculate and print accuracy score
            valid_accuracy = accuracy_score(y_true, y_pred)
            print(
                "Epoch {}: train loss = {}, valid loss = {}, valid acc = {}".format(
                    epoch, training_loss, valid_loss, valid_accuracy
                )
            )
            metrics["training_loss"][model_name].append(training_loss)
            metrics["validation_loss"][model_name].append(valid_loss)
            metrics["validation_accuracy"][model_name].append(valid_accuracy)

            if valid_accuracy > best_val_acc:
                best_val_acc = valid_accuracy
                best_model_name = model_name

            if early_stopping.stop(valid_loss, model):
                print("early stopping...")
                model_weights = torch.load(checkpoint_filename, weights_only=False)
                break

    return best_model_name, best_val_acc, metrics, models

def master_train(train_dataloader, val_dataloader, criterion):
    global use_early_stopping

    input_dim = train_dataloader.dataset.feats.shape[-1]

    use_early_stopping = False
    models = {
        "SimpleLSTM_No_early": BasicLSTM(input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=0),
        "RegLSTM_No_early": BasicLSTM(input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=dropout)
    }

    # Initialize AdamW
    optimizers = {
        "SimpleAdam_No_early": torch.optim.AdamW(models["SimpleLSTM_No_early"].parameters(), lr=lr, weight_decay=0.0),
        "RegAdam_No_early": torch.optim.AdamW(models["RegLSTM_No_early"].parameters(), lr=lr, weight_decay=weight_decay)
    }
    no_early_best_model_name, no_early_best_acc, no_early_stopping_metrics, no_early_models = train(models, optimizers, train_dataloader, val_dataloader, criterion)

    use_early_stopping = True
    models = {
        "RegLSTM_Early": BasicLSTM(input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=dropout),
        "BiLSTM_Early": BasicLSTM(input_dim, rnn_size, output_dim, num_layers, bidirectional=True, dropout=dropout)
    }
    optimizers = {
        "RegAdam_Early": torch.optim.AdamW(models["RegLSTM_Early"].parameters(), lr=lr, weight_decay=weight_decay),
        "BiAdam_Early": torch.optim.AdamW(models["BiLSTM_Early"].parameters(), lr=lr, weight_decay=weight_decay)
    }
    early_best_model_name, early_best_acc, early_stopping_metrics, early_models = train(models, optimizers, train_dataloader, val_dataloader, criterion)

    models = no_early_models | early_models
    metrics = {
        "training_loss": {},
        "validation_loss": {},
        "validation_accuracy": {},
    }
    for metric in metrics.keys():
        metrics[metric] = no_early_stopping_metrics[metric] | early_stopping_metrics[metric]
    
    best_model_name = no_early_best_model_name
    if(early_best_acc > no_early_best_acc):
        best_model_name = early_best_model_name
        
    return best_model_name, metrics, models

def visualize_train(metrics):
    for metric_name in metrics.keys():
        plt.figure(figsize=(8, 6))
        for model_name, values in metrics[metric_name].items():
            plt.plot(range(1, len(values) + 1), values, marker='o', label=model_name)

        plt.title(f"{metric_name.replace('_', ' ').title()} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{metric_name}_plot.png")
        plt.show()

def visualize_train_val_loss(best_model_name, metrics):
    train_loss = metrics["training_loss"].get(best_model_name, [])
    val_loss = metrics["validation_loss"].get(best_model_name, [])

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss", marker='o')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", marker='s')
    
    plt.title(f"Training vs Validation Loss ({best_model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{best_model_name}_train_val_loss.png")
    plt.show()

def test_packing():
    global use_early_stopping, use_packing

    input_dim = train_dataloader.dataset.feats.shape[-1]
    use_early_stopping = False

    use_packing = False
    models = {
        "SimpleLSTM_No_early": BasicLSTM(input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=0),
    }
    optimizers = {
        "SimpleAdam_No_early": torch.optim.AdamW(models["SimpleLSTM_No_early"].parameters(), lr=lr, weight_decay=0.0),
    }
    t1 = time.time()
    train(models, optimizers, train_dataloader, val_dataloader, criterion)
    t2 = time.time()
    duration1 = t2 - t1

    use_packing = True
    models = {
        "SimpleLSTM_No_early": BasicLSTM(input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=0),
    }
    optimizers = {
        "SimpleAdam_No_early": torch.optim.AdamW(models["SimpleLSTM_No_early"].parameters(), lr=lr, weight_decay=0.0),
    }
    t1 = time.time()
    train(models, optimizers, train_dataloader, val_dataloader, criterion)
    t2 = time.time()
    duration2 = t2 - t1
    
    print(f"Model training time without packing: {duration1}.")
    print(f"Model training time with packing: {duration2}.")


if __name__ == '__main__':
    # # Step 9
    # train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
    # labels = list(set(y_train))

    # # Step 10 - 11 -12 -13
    # best_metrics, metrics, best_model = hmm_tuning(4, 5)

    # plot_tuning_results(metrics, best_metrics)

    # print()

    # pred_test, true_test = evaluate(best_model, test_dic, labels)
    # test_accuracy = accuracy_score(true_test, pred_test)
    # cm = confusion_matrix(true_test, pred_test)
    # plot_confusion_matrix(cm, classes=range(10))
    # print()
    # print(f"Accuracy score on the test set is: {test_accuracy}")
    
    # Step 14
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)

    criterion = nn.CrossEntropyLoss()
    best_model_name, metrics, models = master_train(train_dataloader, val_dataloader, criterion)
    visualize_train(metrics)

    test_packing()

    print(f"The best model was {best_model_name}")

    val_loss, y_pred_val, y_true_val = evaluation_loop(models[best_model_name], val_dataloader, criterion)
    test_loss, y_pred_test, y_true_test = evaluation_loop(models[best_model_name], test_dataloader, criterion)

    cm_test = confusion_matrix(y_true_test, y_pred_test)
    plot_confusion_matrix(cm_test, range(10))

    visualize_train_val_loss(best_model_name, metrics)