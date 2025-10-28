import librosa
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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

def analyze_digits_statistics(digits):
    digits_array = np.array(digits)
    
    digit_counter = Counter(digits_array)
    
    total_samples = len(digits_array)
    
    print("=" * 60)
    print("ΣΤΑΤΙΣΤΙΚΑ ΨΗΦΙΩΝ ΑΠΟ ΤΑ ΔΕΔΟΜΕΝΑ")
    print("=" * 60)
    
    unique_digits = sorted(digit_counter.keys())
    
    for digit in unique_digits:
        count = digit_counter[digit]
        percentage = (count / total_samples) * 100
        print(f"Ψηφίο '{digit}': {count:3d} φορές ({percentage:6.2f}%)")
    
    print("=" * 60)
    print(f"Σύνολο δειγμάτων: {total_samples}")
    print(f"Μοναδικά ψηφία: {len(unique_digits)}")
    print(f"Πιο συχνό ψηφίο: {digit_counter.most_common(1)[0][0]} ({digit_counter.most_common(1)[0][1]} φορές)")
    print("=" * 60)
    print()
    
    return digit_counter, total_samples

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

def extract_mfscs(wavs, mfccs, digits, speakers): # Forward
    mfcc_n1_0, mfcc_n1_1, mfcc_n2_0, mfcc_n2_1 = None, None, None, None
    mfsc_n1_0, mfsc_n1_1, mfsc_n2_0, mfsc_n2_1 = None, None, None, None
    speaker_n1_0, speaker_n1_1, speaker_n2_0, speaker_n2_1 = None, None, None, None

    win_len = 25 * Fs // 1000 
    step = 10 * Fs // 1000
    n_features = 13

    for i, wav in enumerate(wavs):
        if digits[i] == n1: 
            if speaker_n1_0 == None:
                mfcc_n1_0 = mfccs[i] 
                mfsc_n1_0 = librosa.feature.melspectrogram(y=wav, sr=Fs, win_length=win_len, hop_length=step, n_mels=n_features)
                speaker_n1_0 = speakers[i]
            elif speaker_n1_1 == None:
                mfcc_n1_1 = mfccs[i]
                mfsc_n1_1 = librosa.feature.melspectrogram(y=wav, sr=Fs, win_length=win_len, hop_length=step, n_mels=n_features)
                speaker_n1_1 = speakers[i]
        elif digits[i] == n2: 
            if speaker_n2_0 == None: 
                mfcc_n2_0 = mfccs[i] 
                mfsc_n2_0 = librosa.feature.melspectrogram(y=wav, sr=Fs, win_length=win_len, hop_length=step, n_mels=n_features)
                speaker_n2_0 = speakers[i]
            elif speaker_n2_1 == None:
                mfcc_n2_1 = mfccs[i] 
                mfsc_n2_1 = librosa.feature.melspectrogram(y=wav, sr=Fs, win_length=win_len, hop_length=step, n_mels=n_features)
                speaker_n2_1 = speakers[i]

        if speaker_n1_0 != None and speaker_n1_1 != None and speaker_n2_0 != None and speaker_n2_1 != None:
            break      

    cor_matrices_mfscs = [np.corrcoef(mfsc) for mfsc in [mfsc_n1_0, mfsc_n1_1, mfsc_n2_0, mfsc_n2_1]]
    cor_matrices_mfccs = [np.corrcoef(mfcc) for mfcc in [mfcc_n1_0, mfcc_n1_1, mfcc_n2_0, mfcc_n2_1]]
    mask = np.triu(np.ones_like(cor_matrices_mfccs[0], dtype=bool))

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize = (10, 10))

    for i in range(4):
        sns.heatmap(cor_matrices_mfscs[i%4], ax=axs[i//2][i%2], cmap="coolwarm", mask=mask, cbar=True, vmin=-1, vmax=1, annot=True, fmt=".2f")
    
    axs[0][0].set_title(r"1\textsuperscript{st} speaker's MFSCs correlation matrix for digit n\textsubscript{1}=2")
    axs[0][1].set_title(r"2\textsuperscript{nd} speaker's MFSCs correlation matrix for digit n\textsubscript{1}=2")
    axs[1][0].set_title(r"1\textsuperscript{st} speaker's MFSCs correlation matrix for digit n\textsubscript{2}=5")
    axs[1][1].set_title(r"2\textsuperscript{nd} speaker's MFSCs correlation matrix for digit n\textsubscript{2}=5")

    fig.suptitle(r"\textbf{1\textsuperscript{st} and 2\textsuperscript{nd} speaker's MFSCs correlation matrix for digits n\textsubscript{1}=2 and n\textsubscript{2}=5}", fontsize=16)

    plt.savefig("plots/corrs_mfsc_step4.png")
    plt.show()

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True, figsize = (10, 10))

    for i in range(4):
        sns.heatmap(cor_matrices_mfccs[i%4], ax=axs[i//2][i%2], cmap="coolwarm", mask=mask, cbar=True, vmin=-1, vmax=1, annot=True, fmt=".2f")

    axs[0][0].set_title(r"1\textsuperscript{st} speaker's MFCCs correlation matrix for digit n\textsubscript{1}=2")
    axs[0][1].set_title(r"2\textsuperscript{nd} speaker's MFCCs correlation matrix for digit n\textsubscript{1}=2")
    axs[1][0].set_title(r"1\textsuperscript{st} speaker's MFCCs correlation matrix for digit n\textsubscript{2}=5")
    axs[1][1].set_title(r"2\textsuperscript{nd} speaker's MFCCs correlation matrix for digit n\textsubscript{2}=5")

    fig.suptitle(r"\textbf{1\textsuperscript{st} and 2\textsuperscript{nd} speaker's MFCCs correlation matrix for digits n\textsubscript{1}=2 and n\textsubscript{2}=5}", fontsize=16)

    plt.savefig("plots/corrs_mfcc_step4.png")
    plt.show()

def combine_features(mfccs, deltas, delta_deltas, digits):
    statistical_characteristics = []

    for mfcc, delta, delta_delta in zip(mfccs, deltas, delta_deltas):
        features = np.vstack((mfcc, delta, delta_delta))   # shape -> (39, smth)

        mean = np.mean(features, axis=1) # axis=1 -> (39,)
        std = np.std(features, axis=1)

        characteristics = np.concatenate((mean, std)) # (78,)
        statistical_characteristics.append(characteristics)

    feature_vectors = np.array(statistical_characteristics)  # (133, 78)

    unique_digits = np.array(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
    markers = ['o','s','D','^','v','P','X','*','<','>']
    colors = plt.cm.Set1.colors

    plt.figure(figsize=(10,10))

    for i, d in enumerate(unique_digits):
        mask = np.array(digits) == d
        plt.scatter(feature_vectors[mask,0], feature_vectors[mask,1], 
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    edgecolor='k', alpha=0.8, s=60, label=str(d))
    plt.xlabel(r'Mean MFCC1')
    plt.ylabel(r'Mean MFCC2')
    plt.title(r'Scatter plot of first 2 dimensions per digit')
    plt.grid(True)
    plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/scatter_plot_2_random_features_step5.png")
    plt.show()
    
    return feature_vectors

def pca_on_features(feature_vectors, digits):
    pca_2dims = PCA(n_components=2)
    X2d = pca_2dims.fit_transform(feature_vectors)
    print(pca_2dims.explained_variance_ratio_)
    print()
    
    unique_digits = np.array(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
    markers = ['o','s','D','^','v','P','X','*','<','>']
    colors = plt.cm.Set1.colors

    plt.figure(figsize=(10,10))

    for i, d in enumerate(unique_digits):
        mask = np.array(digits) == d
        plt.scatter(X2d[mask,0], X2d[mask,1], 
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    edgecolor='k', alpha=0.8, s=60, label=str(d))
    plt.xlabel(r'Coefficient in 1\textsuperscript{st} eigendirection')
    plt.ylabel(r'Coefficient in 2\textsuperscript{nd} eigendirection')
    plt.title(r'Scatter plot of first 2 principal components per digit', fontsize=16)
    plt.grid(True)
    plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/scatter_plot_2_eigendirections_step6.png")
    plt.show()
    
    pca_3dims = PCA(n_components=3)
    X3d = pca_3dims.fit_transform(feature_vectors)
    print(pca_3dims.explained_variance_ratio_)
    print()

    unique_digits = np.array(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', '<', '>']
    colors = plt.cm.Set1.colors

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    for i, d in enumerate(unique_digits):
        mask = np.array(digits) == d
        ax.scatter(X3d[mask, 0], X3d[mask, 1], X3d[mask, 2],
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                edgecolor='k', alpha=0.85, s=55, label=str(d))

    ax.set_xlabel('1st Principal Component', labelpad=10, fontsize=11)
    ax.set_ylabel('2nd Principal Component', labelpad=10, fontsize=11)
    ax.set_zlabel('3rd Principal Component', labelpad=10, fontsize=11)

    ax.set_title('Scatter Plot of First 3 Principal Components per Digit',
                fontsize=18, fontweight='bold', pad=20, loc='center')

    ax.view_init(elev=25, azim=35)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(title='Digit', loc='upper left', bbox_to_anchor=(0.02, 0.98))

    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    plt.savefig("plots/scatter_plot_3_eigendirections_step6.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    return X2d, X3d

def classify_Naive_Bayes(X, digits):
    accuracies = []
    f1s = []
    
    for n in range(1, 11):
        X_pca = X
        if n != 1:
            X_pca = PCA(n_components=n).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_pca, digits, test_size=0.3, random_state=42)
        gnb = GaussianNB()

        y_pred = gnb.fit(X_train, y_train).predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        if n != 1:
            print(f"PCA dimensions: {n}")
            accuracies.append(accuracy)
            f1s.append(f1)
        else:
            print("Without PCA:")
            
        print(f"Accuracy: {accuracy}")
        print(f"F1 score: {f1}")
        print()

    ns = np.arange(2, 11, 1)
    accuracies = np.array(accuracies)
    f1s = np.array(f1s)

    plt.plot(ns, accuracies, label="Accuracy", marker="*")
    plt.plot(ns, f1s, label="F1-score", marker="o")
    plt.xlabel("PCA dimensions")
    plt.title("Accuracy and F1-score vs PCA dimensions for Naive Bayes Classifier")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("plots/accuracies_and_f1_naive_bayes_step7.png")
    plt.show()
    
def classify_SVM(X, digits):
    accuracies = []
    f1s = []
    
    for n in range(1, 11):
        X_pca = X

        if n != 1:
            X_pca = PCA(n_components=n).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_pca, digits, test_size=0.3, random_state=42)
        svm = SVC()

        y_pred = svm.fit(X_train, y_train).predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        if n != 1:
            print(f"PCA dimensions: {n}")
            accuracies.append(accuracy)
            f1s.append(f1)
        else:
            print("Without PCA:")
            
        print(f"Accuracy: {accuracy}")
        print(f"F1 score: {f1}")
        print()

    ns = np.arange(2, 11, 1)
    accuracies = np.array(accuracies)
    f1s = np.array(f1s)

    plt.plot(ns, accuracies, label="Accuracy", marker="*")
    plt.plot(ns, f1s, label="F1-score", marker="o")
    plt.xlabel("PCA dimensions")
    plt.title("Accuracy and F1-score vs PCA dimensions for SVM classifier")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("plots/accuracies_and_f1_svm_step7.png")
    plt.show()

    # Find best configurations
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1s)

    print("="*60)
    print(" Optimal Configuration (by Accuracy)")
    print(f"  PCA dimensions = {ns[best_acc_idx]}")
    print(f"  Accuracy = {accuracies[best_acc_idx]:.4f}")
    print(f"  F1 score = {f1s[best_acc_idx]:.4f}")
    print("="*60)
    print(" Optimal Configuration (by F1-score)")
    print(f"  PCA dimensions = {ns[best_f1_idx]}")
    print(f"  F1 score = {f1s[best_f1_idx]:.4f}")
    print(f"  Accuracy = {accuracies[best_f1_idx]:.4f}")
    print("="*60)

def classify_KNN(X, digits):
    ks, ns, accuracies, f1s = [], [], [], []
    
    for k in range(1, 8, 2):
        for n in range(2, 11):
            X_pca = PCA(n_components=n).fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_pca, digits, test_size=0.3, random_state=42)
            neigh = KNeighborsClassifier(n_neighbors=k)

            y_pred = neigh.fit(X_train, y_train).predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"Number of neighbors: {k}")
            print(f"PCA dimensions: {n}")
            accuracies.append(accuracy)
            f1s.append(f1)
            ks.append(k)
            ns.append(n)
            
            print(f"Accuracy: {accuracy}")
            print(f"F1 score: {f1}")
            print()

    ks = np.array(ks)
    ns = np.array(ns)
    accuracies = np.array(accuracies)
    f1s = np.array(f1s)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ns, ks, accuracies, c='b', marker='o', label='Accuracy', s=50)
    ax.scatter(ns, ks, f1s, c='r', marker='^', label='F1-score', s=50)

    ax.set_xlabel('PCA Dimensions')
    ax.set_ylabel('k (Neighbors)')
    ax.set_zlabel('Score')
    ax.set_title('3D Plot: Accuracy and F1-score vs PCA dimensions and k for KNN Classifier')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("plots/3D_accuracy_f1_knn_step7.png", dpi=300)
    plt.show()

    # Find best configurations
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1s)

    print("="*60)
    print(" Optimal Configuration (by Accuracy)")
    print(f"  k = {ks[best_acc_idx]}")
    print(f"  PCA dimensions = {ns[best_acc_idx]}")
    print(f"  Accuracy = {accuracies[best_acc_idx]:.4f}")
    print(f"  F1 score = {f1s[best_acc_idx]:.4f}")
    print("="*60)
    print("  Optimal Configuration (by F1-score)")
    print(f"  k = {ks[best_f1_idx]}")
    print(f"  PCA dimensions = {ns[best_f1_idx]}")
    print(f"  F1 score = {f1s[best_f1_idx]:.4f}")
    print(f"  Accuracy = {accuracies[best_f1_idx]:.4f}")
    print("="*60)

def classify_GBC(X, digits):
    ks, ns, accuracies, f1s = [], [], [], []
    
    for k in range(3, 10, 2):
        for n in range(2, 11):
            X_pca = PCA(n_components=n).fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_pca, digits, test_size=0.3, random_state=42)
            gbc = GradientBoostingClassifier(max_depth=k, random_state=42)

            y_pred = gbc.fit(X_train, y_train).predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"Max  depth: {k}")
            print(f"PCA dimensions: {n}")
            accuracies.append(accuracy)
            f1s.append(f1)
            ks.append(k)
            ns.append(n)
            
            print(f"Accuracy: {accuracy}")
            print(f"F1 score: {f1}")
            print()

    ks = np.array(ks)
    ns = np.array(ns)
    accuracies = np.array(accuracies)
    f1s = np.array(f1s)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ns, ks, accuracies, c='b', marker='o', label='Accuracy', s=50)
    ax.scatter(ns, ks, f1s, c='r', marker='^', label='F1-score', s=50)

    ax.set_xlabel('PCA Dimensions')
    ax.set_ylabel('k (Max Depth)')
    ax.set_zlabel('Score')
    ax.set_title('3D Plot: Accuracy and F1-score vs PCA dimensions and k for Gradient Boosting Classifier')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("plots/3D_accuracy_f1_gb_step7.png", dpi=300)
    plt.show()

    # Find best configurations
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1s)

    print("="*60)
    print(" Optimal Configuration (by Accuracy)")
    print(f"  max depth = {ks[best_acc_idx]}")
    print(f"  PCA dimensions = {ns[best_acc_idx]}")
    print(f"  Accuracy = {accuracies[best_acc_idx]:.4f}")
    print(f"  F1 score = {f1s[best_acc_idx]:.4f}")
    print("="*60)
    print("  Optimal Configuration (by F1-score)")
    print(f"  max depth = {ks[best_f1_idx]}")
    print(f"  PCA dimensions = {ns[best_f1_idx]}")
    print(f"  F1 score = {f1s[best_f1_idx]:.4f}")
    print(f"  Accuracy = {accuracies[best_f1_idx]:.4f}")
    print("="*60)

def classify_LR(X, digits):
    accuracies = []
    f1s = []
    
    for n in range(2, 11):
        X_pca = PCA(n_components=n).fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_pca, digits, test_size=0.3, random_state=42)
        lr = LogisticRegression(random_state=42)

        y_pred = lr.fit(X_train, y_train).predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"PCA dimensions: {n}")
        accuracies.append(accuracy)
        f1s.append(f1)
            
        print(f"Accuracy: {accuracy}")
        print(f"F1 score: {f1}")
        print()

    ns = np.arange(2, 11, 1)
    accuracies = np.array(accuracies)
    f1s = np.array(f1s)

    plt.plot(ns, accuracies, label="Accuracy", marker="*")
    plt.plot(ns, f1s, label="F1-score", marker="o")
    plt.xlabel("PCA dimensions")
    plt.title("Accuracy and F1-score vs PCA dimensions for Logistic Regression")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("plots/accuracies_and_f1_lr_step7.png")
    plt.show()

    # Find best configurations
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1s)

    print("="*60)
    print(" Optimal Configuration (by Accuracy)")
    print(f"  PCA dimensions = {ns[best_acc_idx]}")
    print(f"  Accuracy = {accuracies[best_acc_idx]:.4f}")
    print(f"  F1 score = {f1s[best_acc_idx]:.4f}")
    print("="*60)
    print(" Optimal Configuration (by F1-score)")
    print(f"  PCA dimensions = {ns[best_f1_idx]}")
    print(f"  F1 score = {f1s[best_f1_idx]:.4f}")
    print(f"  Accuracy = {accuracies[best_f1_idx]:.4f}")
    print("="*60)



class myRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(myRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


class myLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1): # hidden_size = 32 for moderate expressiveness
        super(myLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # input is in format (batch_size, seq_len, input_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


class myGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(myGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


def EvalCompNNs():
    f = 40 # f = 40Hz
    total_points = 20*f
    t = np.linspace(0, 1, total_points) # 1 second, create 40 points for each period
    x = np.sin(2*np.pi*f*t) # create input sine wave
    y = np.cos(2*np.pi*f*t) # create target cosine wave

    seq_len = 10 # sequence length is 10
    X, Y = [], []
    for i in range(total_points - seq_len):
        X.append(x[i:i+seq_len])
        Y.append(y[i:i+seq_len])

    # Use CUDA for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # convert to torch tensors and add extra dimension of size 1 (1 input/output at each time step)
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1).to(device)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1).to(device)
    
    models = {
        'RNN': myRNN().to(device),
        'LSTM': myLSTM().to(device),
        'GRU': myGRU().to(device)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n====== Trainning {model_name} ======")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # we choose a small learning rate to avoid overshooting
        
        epochs = 100
        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad() # clear previous gradients
            out = model(X)
            loss = criterion(out, Y) # we use MSE loss function for regression
            loss.backward() # backpropagate the loss gradients
            optimizer.step() # update the weights
            losses.append(loss.item())
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

        results[model_name] = {
            'model': model,
            'losses': losses,
            'final_loss': losses[-1]
        }

    # Plot training loss
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    for model_name in models.keys():
        plt.plot(range(1, epochs + 1), results[model_name]['losses'], 
                label=f'{model_name} (Final: {results[model_name]["final_loss"]:.6f})', 
                linewidth=2)
    
    plt.title("Training Loss Comparison: RNN vs LSTM vs GRU")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    #plt.yscale('log') 
    plt.tight_layout()



    plt.subplot(2, 1, 2)
    actual = Y.cpu().numpy()[:, -1, 0]

    plt.plot(actual, label="Actual Cosine", color='black', linewidth=3, alpha=0.7)

    colors = ['red', 'blue', 'green']
    for i, (model_name, result) in enumerate(results.items()):
        model = result['model']
        model.eval()
        with torch.no_grad():
            predictions = model(X).cpu().numpy()
        
        predicted = predictions[:, -1, 0]
        plt.plot(predicted, label=f'Predicted {model_name}', 
                linestyle='--', linewidth=2, alpha=0.8, color=colors[i])

    plt.title("Predicted vs Actual Cosine Wave (All Models)")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("plots/all_models_comparison_step8.png", dpi=300)
    plt.show()

    actual = Y.cpu().numpy()[:, -1, 0]
    
    for model_name, result in results.items():
        model = result['model']
        model.eval()
        with torch.no_grad():
            predictions = model(X).cpu().numpy()
        
        predicted = predictions[:, -1, 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(range(1, epochs + 1), result['losses'], 'b-', linewidth=2)
        ax1.set_title(f"{model_name} - Training Loss\nFinal: {result['final_loss']:.6f}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE Loss")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        time_steps = np.arange(len(actual))
        ax2.plot(time_steps, actual, label="Actual Cosine", 
                color='black', linewidth=2, alpha=0.8)
        ax2.plot(time_steps, predicted, label=f"Predicted {model_name}", 
                linestyle="--", linewidth=2, alpha=0.9)
        ax2.set_title(f"{model_name} - Predictions vs Actual")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if len(actual) > 100:
            zoom_start = 100
            zoom_end = 200
            ax2.set_xlim(zoom_start, zoom_end)
            ax2.set_title(f"{model_name} - Predictions vs Actual\n(Zoomed Section 100-200)")
        
        plt.tight_layout()
        plt.savefig(f"plots/{model_name.lower()}_clean.png", dpi=300, bbox_inches='tight')
        plt.show()

    print("\n" + "="*50)
    print("FINAL COMPARISON RESULTS:")
    print("="*50)
    for model_name, result in results.items():
        model = result['model']
        model.eval()
        with torch.no_grad():
            predictions = model(X).cpu().numpy()
        predicted = predictions[:, -1, 0]
        error = np.abs(actual - predicted)
        
        print(f"{model_name}:")
        print(f"  Final Loss: {result['final_loss']:.6f}")
        print(f"  Mean Absolute Error: {np.mean(error):.6f}")
        print()


if __name__ == '__main__':
    # Step 2
    wavs, speakers, digits = data_parser()
    analyze_digits_statistics(digits)

    # Step 3
    mfccs, deltas, delta_deltas = extract_features(wavs)
    
    # Step 4.1
    plot_histgrams(wavs, digits, mfccs)

    # Step 4.2
    extract_mfscs(wavs, mfccs, digits, speakers)

    # mfscs1 = extract_mfscs1(mfccs)
    # mfscs2 = extract_mfscs2(mfccs)

    # Step 5
    feature_vectors = combine_features(mfccs, deltas, delta_deltas, digits)

    # Step 6
    pca2d, pca3d = pca_on_features(feature_vectors, digits)

    # Step 7
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_vectors)
    
    # classify_Naive_Bayes(X, digits)
    # classify_SVM(X, digits)
    # classify_KNN(X, digits)
    # classify_GBC(X, digits)
    # classify_LR(X, digits)

    # Step 8
    EvalCompNNs()