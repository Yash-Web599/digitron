"""
This module contains functions for feature extraction and engineering for the
preparatory analysis (Steps 1-7).
"""
from typing import List

import librosa
import numpy as np
from tqdm import tqdm


def extract_mfcc_features(
    wav_list: List[np.ndarray],
    sampling_rate: int = 16000,
    n_mfcc: int = 13,
    window_ms: int = 25,
    hop_ms: int = 10,
) -> List[np.ndarray]:
    """
    Extracts a comprehensive set of features (MFCC, Delta, Delta-Delta)
    from a list of audio waveforms.

    Args:
        wav_list (List[np.ndarray]): A list of numpy arrays (audio waveforms).
        sampling_rate (int): The sampling rate of the audio files.
        n_mfcc (int): The number of base MFCCs to compute.
        window_ms (int): The length of the analysis window in milliseconds.
        hop_ms (int): The step size (hop length) between frames in milliseconds.

    Returns:
        List[np.ndarray]: A list of numpy arrays. Each array has the shape
                          (number_of_frames, 3 * n_mfcc), representing the
                          full features for one audio file.
    """
    print(f"Extracting features for {len(wav_list)} audio files...")

    # Convert time-based parameters to sample-based
    n_fft = int((window_ms / 1000) * sampling_rate)
    hop_length = int((hop_ms / 1000) * sampling_rate)

    full_feature_list = []

    for wav in tqdm(wav_list, desc="Extracting MFCC features"):
        # Calculate Base MFCCs
        mfccs = librosa.feature.mfcc(
            y=wav, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        ).T
        # Calculate Delta and Delta-Delta Features
        delta_mfccs = librosa.feature.delta(mfccs, order=1)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # Concatenate All Features
        full_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=1)
        full_feature_list.append(full_features)

    print("-> Feature extraction complete.")
    return full_feature_list


def create_utterance_vectors(feature_list: List[np.ndarray]) -> np.ndarray:
    """
    Summarizes frame-level features into a single feature vector per utterance.

    For each feature matrix, this function computes the mean and standard
    deviation across all frames for each feature dimension and concatenates them.

    Args:
        feature_list (List[np.ndarray]): A list of frame-level feature arrays.

    Returns:
        np.ndarray: A 2D numpy array of shape (num_utterances, 2 * num_features),
                    where each row is a summarized feature vector.
    """
    print(f"Summarizing features for {len(feature_list)} utterances...")
    utterance_vectors = []

    # Iterate through each feature matrix (one per audio file)
    for feature_matrix in tqdm(feature_list, desc="Creating utterance vectors"):
        # Calculate the mean for each of the 39 features along the time axis (axis=0)
        mean_features = np.mean(feature_matrix, axis=0)
        # Calculate the standard deviation for each of the 39 features
        std_features = np.std(feature_matrix, axis=0)
        # The final vector has 39 (mean) + 39 (std) = 78 dimensions
        utterance_vector = np.concatenate((mean_features, std_features))
        utterance_vectors.append(utterance_vector)

    print("-> Utterance vector creation complete.")
    return np.array(utterance_vectors)


def add_bonus_features(
    wav_list: List[np.ndarray], base_vectors: np.ndarray, sr: int = 16000
) -> np.ndarray:
    """
    Calculates bonus features (ZCR, Spectral Centroid) and appends them
    to the base feature vectors.

    Args:
        wav_list (List[np.ndarray]): The list of original audio waveforms.
        base_vectors (np.ndarray): The existing utterance vectors (e.g., from MFCCs).
        sr (int): The sampling rate of the audio.

    Returns:
        np.ndarray: The extended feature vectors with the new features appended.
    """
    print("Engineering bonus features...")
    bonus_features_list = []
    for wav in tqdm(wav_list, desc="Calculating ZCR & Spectral Centroid"):
        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=wav)[0]
        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=wav, sr=sr)[0]

        # Summarize with mean and std, creating 4 new features
        bonus_vec = [
            np.mean(zcr),
            np.std(zcr),
            np.mean(spec_cent),
            np.std(spec_cent),
        ]
        bonus_features_list.append(bonus_vec)

    bonus_features_array = np.array(bonus_features_list)
    extended_vectors = np.concatenate((base_vectors, bonus_features_array), axis=1)

    print(f"-> New feature vectors created with shape: {extended_vectors.shape}")
    return extended_vectors