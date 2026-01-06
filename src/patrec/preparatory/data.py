"""
This module contains functions for loading and parsing the initial 'digits'
dataset used in the preparatory analysis (Steps 1-7).
"""
import os
from glob import glob
from typing import List, Tuple

import librosa
import numpy as np
from tqdm import tqdm


def parse_digit_recordings(
    directory_path: str,
) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Parses a directory of isolated digit recordings.

    This function reads all .wav files from the specified directory. It assumes
    the filename format is '<digit_name><speaker_id>.wav' (e.g., 'eight8.wav').
    It extracts the waveform, speaker ID, and the spoken digit for each file.

    Args:
        directory_path (str): The path to the directory containing the .wav files.

    Returns:
        Tuple[List[np.ndarray], List[str], List[int]]: A tuple containing three lists:
            - wav_data: A list of numpy arrays, where each array is the audio waveform.
            - speakers: A list of strings, where each string is the speaker's ID.
            - digits: A list of integers, where each integer is the spoken digit.
    """
    # --- Initialization ---
    wav_data = []
    speakers = []
    digits = []

    # A mapping to convert the written digit name (string) to an integer.
    digit_name_to_number = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    }

    # --- Find and Process All .wav Files ---
    search_pattern = os.path.join(directory_path, "*.wav")
    file_paths = sorted(glob(search_pattern))

    if not file_paths:
        raise FileNotFoundError(
            f"No .wav files found in the specified directory: {directory_path}"
        )

    print(f"Found {len(file_paths)} .wav files in '{directory_path}'. Processing...")

    for file_path in tqdm(file_paths, desc="Parsing audio files"):
        # sr=None ensures we load the file with its original sampling rate (16kHz).
        wav, _ = librosa.load(file_path, sr=None)
        wav_data.append(wav)

        # --- Parse Filename to Extract Speaker and Digit ---
        basename = os.path.basename(file_path)
        filename_stem = os.path.splitext(basename)[0]

        digit_name = ""
        speaker_id = ""
        for i, char in enumerate(filename_stem):
            if char.isdigit():
                digit_name = filename_stem[:i]
                speaker_id = filename_stem[i:]
                break

        speakers.append(speaker_id)
        digits.append(digit_name_to_number[digit_name])

    print("-> Audio data parsing complete.")
    return wav_data, speakers, digits