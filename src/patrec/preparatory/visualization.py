"""
This module contains all visualization functions for the preparatory analysis,
including histograms, heatmaps, and scatter plots for feature exploration.
"""
from typing import List, Dict, Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def plot_mfcc_histograms(
    all_features: List[np.ndarray], all_digits: List[int], n1: int, n2: int, output_path: str = None
):
    """
    Plots the histograms of the 1st and 2nd MFCC coefficients for two specified digits.

    Args:
        all_features (List[np.ndarray]): The list of all frame-level feature matrices.
        all_digits (List[int]): The list of corresponding digits for each feature matrix.
        n1 (int): The first digit to analyze.
        n2 (int): The second digit to analyze.
        output_path (str, optional): If provided, saves the plot to this path. Defaults to None.
    """
    # Collect all frames for each of the two target digits.
    # The first MFCC is at column index 0, the second is at index 1.
    mfcc1_digit_n1, mfcc2_digit_n1 = [], []
    mfcc1_digit_n2, mfcc2_digit_n2 = [], []

    for features, digit in zip(all_features, all_digits):
        if digit == n1:
            mfcc1_digit_n1.append(features[:, 0])  # All frames, 1st MFCC
            mfcc2_digit_n1.append(features[:, 1])  # All frames, 2nd MFCC
        elif digit == n2:
            mfcc1_digit_n2.append(features[:, 0])  # All frames, 1st MFCC
            mfcc2_digit_n2.append(features[:, 1])  # All frames, 2nd MFCC

    # Concatenate the lists of arrays into single, long numpy arrays for plotting.
    mfcc1_digit_n1 = np.concatenate(mfcc1_digit_n1)
    mfcc2_digit_n1 = np.concatenate(mfcc2_digit_n1)
    mfcc1_digit_n2 = np.concatenate(mfcc1_digit_n2)
    mfcc2_digit_n2 = np.concatenate(mfcc2_digit_n2)

    # Create a figure with two subplots side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Distribution of MFCCs for Digits {n1} and {n2}', fontsize=16)

    # Plot for the 1st MFCC.
    sns.histplot(mfcc1_digit_n1, ax=axes[0], color='skyblue', kde=True, label=f'Digit {n1}')
    sns.histplot(mfcc1_digit_n2, ax=axes[0], color='salmon', kde=True, label=f'Digit {n2}')
    axes[0].set_title('1st MFCC Coefficient')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot for the 2nd MFCC.
    sns.histplot(mfcc2_digit_n1, ax=axes[1], color='skyblue', kde=True, label=f'Digit {n1}')
    sns.histplot(mfcc2_digit_n2, ax=axes[1], color='salmon', kde=True, label=f'Digit {n2}')
    axes[1].set_title('2nd MFCC Coefficient')
    axes[1].set_xlabel('Value')
    axes[1].legend()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"MFCC histograms saved to {output_path}")
    plt.show()


def plot_feature_correlation_heatmaps(
    wav_list: List[np.ndarray], all_features: List[np.ndarray], all_digits: List[int], all_speakers: List[str], n1: int, n2: int, output_path: str = None
):
    """
    Plots the correlation heatmaps for MFSCs vs. MFCCs for specific utterances.

    Args:
        wav_list (List[np.ndarray]): The list of raw audio waveforms.
        all_features (List[np.ndarray]): The list of all frame-level feature matrices.
        all_digits (List[int]): The list of corresponding digits.
        all_speakers (List[str]): The list of corresponding speakers.
        n1 (int): The first digit to select utterances from.
        n2 (int): The second digit to select utterances from.
        output_path (str, optional): If provided, saves the plot to this path. Defaults to None.
    """
    # --- Helper function to extract MFSCs ---
    def _extract_mfscs(wavs, sampling_rate=16000, n_mels=13, window_ms=25, hop_ms=10):
        n_fft = int((window_ms / 1000) * sampling_rate)
        hop_length = int((hop_ms / 1000) * sampling_rate)
        mfsc_list = []
        for wav in wavs:
            mel_spec = librosa.feature.melspectrogram(y=wav, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max).T
            mfsc_list.append(log_mel_spec)
        return mfsc_list

    # --- Find 4 specific utterances to analyze (2 for each digit from different speakers) ---
    utterances_to_analyze = []
    # This logic finds the first two occurrences of n1 and n2 from unique speakers.
    speakers_used_n1 = set()
    speakers_used_n2 = set()
    for i, (digit, speaker) in enumerate(zip(all_digits, all_speakers)):
        if len(speakers_used_n1) < 2 and digit == n1 and speaker not in speakers_used_n1:
            utterances_to_analyze.append({'index': i, 'digit': digit, 'speaker': speaker})
            speakers_used_n1.add(speaker)
        if len(speakers_used_n2) < 2 and digit == n2 and speaker not in speakers_used_n2:
            utterances_to_analyze.append({'index': i, 'digit': digit, 'speaker': speaker})
            speakers_used_n2.add(speaker)
        if len(utterances_to_analyze) == 4:
            break

    # --- Extract the necessary data for plotting ---
    wavs_to_analyze = [wav_list[utt['index']] for utt in utterances_to_analyze]
    mfccs_list = [all_features[utt['index']][:, :13] for utt in utterances_to_analyze]
    mfscs_list = _extract_mfscs(wavs_to_analyze)

    # --- Plot the Correlation Heatmaps ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Feature Correlation: MFSC (top) vs. MFCC (bottom)', fontsize=16)

    for i, utt in enumerate(utterances_to_analyze):
        # Plot MFSC Correlation
        corr_mfsc = np.corrcoef(mfscs_list[i].T)
        sns.heatmap(corr_mfsc, ax=axes[0, i], cmap='vlag', vmin=-1, vmax=1)
        axes[0, i].set_title(f"MFSC: Digit {utt['digit']}, Spk {utt['speaker']}")
        axes[0, i].set_xlabel("Feature Index")
        axes[0, i].set_ylabel("Feature Index")

        # Plot MFCC Correlation
        corr_mfcc = np.corrcoef(mfccs_list[i].T)
        sns.heatmap(corr_mfcc, ax=axes[1, i], cmap='vlag', vmin=-1, vmax=1)
        axes[1, i].set_title(f"MFCC: Digit {utt['digit']}, Spk {utt['speaker']}")
        axes[1, i].set_xlabel("Feature Index")
        axes[1, i].set_ylabel("Feature Index")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Feature correlation heatmaps saved to {output_path}")
    plt.show()


def plot_pca_scatter(
    X_pca: np.ndarray, y: List[int], total_variance_explained: float, output_path: str = None
):
    """
    Generates a 2D or 3D scatter plot of PCA-transformed data.

    Args:
        X_pca (np.ndarray): The data after PCA transformation (shape: num_samples x 2 or num_samples x 3).
        y (List[int]): The list of labels for coloring the points.
        total_variance_explained (float): The total variance explained by the PCA components.
        output_path (str, optional): If provided, saves the plot to this path. Defaults to None.
    """
    # Determine if it's a 2D or 3D plot based on the shape of the data.
    n_dims = X_pca.shape[1]
    if n_dims not in [2, 3]:
        raise ValueError("This function only supports 2D or 3D PCA plots.")

    # --- Plot setup ---
    fig = plt.figure(figsize=(12, 10))
    title = f'PCA of Utterance Features ({n_dims}D)\n(Explains {total_variance_explained:.2f}% of variance)'

    # --- Color and marker setup ---
    unique_digits = sorted(np.unique(y))
    colors = plt.cm.get_cmap('tab10', len(unique_digits))
    color_map = {digit: colors(i) for i, digit in enumerate(unique_digits)}
    digits_array = np.array(y)

    # --- 2D Plotting Logic ---
    if n_dims == 2:
        ax = fig.add_subplot(111)
        for digit in unique_digits:
            indices = np.where(digits_array == digit)[0]
            ax.scatter(
                X_pca[indices, 0], X_pca[indices, 1],
                color=color_map[digit], marker=f'${digit}$', s=100, label=f'Digit {digit}'
            )
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend(title='Digits', bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    # --- 3D Plotting Logic ---
    else: # n_dims == 3
        ax = fig.add_subplot(111, projection='3d')
        for digit in unique_digits:
            indices = np.where(digits_array == digit)[0]
            ax.scatter(
                X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2],
                color=color_map[digit], marker=f'${digit}$', s=100, label=f'Digit {digit}'
            )
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend(title='Digits')
        ax.view_init(elev=20, azim=45)

    ax.set_title(title)
    ax.grid(True)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"PCA scatter plot saved to {output_path}")
    plt.show()

def plot_classification_results(results_list: List[Dict], unique_labels: List[int], title: str, output_path: str = None):
    """Plots a grid of confusion matrices for a list of classifier results."""
    num_classifiers = len(results_list)
    ncols = 3
    nrows = (num_classifiers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    fig.suptitle(title, fontsize=20)
    axes = axes.flatten()
    for i, result in enumerate(results_list):
        cm = result['confusion_matrix']
        name = result['Classifier']
        accuracy = result['Accuracy']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=unique_labels, yticklabels=unique_labels)
        axes[i].set_title(f"{name}\nAccuracy: {accuracy:.4f}")
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"INFO: Confusion matrices plot saved to {output_path}")
    plt.show()