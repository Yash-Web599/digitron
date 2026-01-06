"""
Main executable script for the Preparatory Analysis (Steps 1-7).

This script performs the following sequence of operations:
1.  Defines constants and configuration parameters.
2.  Creates the output directory for storing plots.
3.  Loads the initial 'digits' dataset using the dedicated parser.
4.  Extracts MFCC, delta, and delta-delta features.
5.  Generates analytical plots (MFCC histograms, feature correlation heatmaps).
6.  Creates utterance-level feature vectors (mean & std).
7.  Applies PCA for dimensionality reduction and visualizes the results.
8.  Runs a systematic classification pipeline on baseline and extended features.
9.  Visualizes the classification results with confusion matrices.
10. Prints a final summary comparing the performance of all classifiers.
"""
import os
import numpy as np
import pandas as pd

from src.patrec.preparatory.data import parse_digit_recordings
from src.patrec.preparatory.features import (
    extract_mfcc_features,
    create_utterance_vectors,
    add_bonus_features,
)
from src.patrec.preparatory.classification import (
    run_classification_pipeline,
    apply_pca,
)
from src.patrec.preparatory.visualization import (
    plot_mfcc_histograms,
    plot_feature_correlation_heatmaps,
    plot_pca_scatter,
    plot_classification_results,
)

# --- 1. Configuration and Constants ---
DATA_DIR = "data/digits"
OUTPUT_DIR = "outputs/preparatory_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N1 = 1
N2 = 2

# --- 2. Data Loading (Step 2) ---
print("--- [Step 2] Loading and Parsing Audio Data ---")
all_wavs, all_speakers, all_digits = parse_digit_recordings(DATA_DIR)

# --- 3. Feature Extraction (Step 3) ---
print("\n--- [Step 3] Extracting MFCC-based Features ---")
all_features = extract_mfcc_features(all_wavs)

# --- 4. Feature Analysis & Visualization (Step 4) ---
print("\n--- [Step 4] Analyzing and Visualizing Features ---")
plot_mfcc_histograms(
    all_features, all_digits, n1=N1, n2=N2, output_path=os.path.join(OUTPUT_DIR, "mfcc_histograms.png")
)
plot_feature_correlation_heatmaps(
    all_wavs, all_features, all_digits, all_speakers, n1=N1, n2=N2, output_path=os.path.join(OUTPUT_DIR, "feature_correlation.png")
)

# --- 5. Utterance-Level Feature Creation (Step 5) ---
print("\n--- [Step 5] Creating Utterance-Level Feature Vectors ---")
utterance_vectors_base = create_utterance_vectors(all_features)

# --- 6. Dimensionality Reduction with PCA (Step 6) ---
print("\n--- [Step 6] Applying PCA for Visualization ---")
from sklearn.model_selection import train_test_split
X_train, X_test, _, _ = train_test_split(utterance_vectors_base, all_digits, test_size=0.3, random_state=42)
_, _, pca_2d = apply_pca(X_train, X_test, n_components=2)
_, _, pca_3d = apply_pca(X_train, X_test, n_components=3)

X_pca2d = pca_2d.transform(utterance_vectors_base)
X_pca3d = pca_3d.transform(utterance_vectors_base)

plot_pca_scatter(
    X_pca2d, all_digits, sum(pca_2d.explained_variance_ratio_) * 100, output_path=os.path.join(OUTPUT_DIR, "pca_2d_scatter.png")
)
plot_pca_scatter(
    X_pca3d, all_digits, sum(pca_3d.explained_variance_ratio_) * 100, output_path=os.path.join(OUTPUT_DIR, "pca_3d_scatter.png")
)

# --- 7. Systematic Classification (Step 7) ---
print("\n--- [Step 7] Running Systematic Classifier Evaluation ---")
unique_labels = sorted(np.unique(all_digits))

# Run for baseline features
results_base_list = run_classification_pipeline(
    X=utterance_vectors_base, y=all_digits, feature_set_name="Baseline (MFCC Mean/Std)"
)
plot_classification_results(
    results_base_list, unique_labels,
    title="Confusion Matrices - Baseline Features",
    output_path=os.path.join(OUTPUT_DIR, "cm_baseline.png")
)

# Run for extended features
print("\n--- [Bonus] Creating and Evaluating Extended Feature Set ---")
utterance_vectors_extended = add_bonus_features(all_wavs, utterance_vectors_base)
results_extended_list = run_classification_pipeline(
    X=utterance_vectors_extended,
    y=all_digits,
    feature_set_name="Extended (MFCC + ZCR/Centroid)",
)
plot_classification_results(
    results_extended_list, unique_labels,
    title="Confusion Matrices - Extended Features",
    output_path=os.path.join(OUTPUT_DIR, "cm_extended.png")
)

# --- 8. Final Summary ---
print("\n" + "=" * 80)
print("FINAL SUMMARY OF CLASSIFIER PERFORMANCE")
print("=" * 80)

df_base = pd.DataFrame(results_base_list)
df_extended = pd.DataFrame(results_extended_list)

summary_df = pd.merge(
    df_base[["Classifier", "Accuracy"]],
    df_extended[["Classifier", "Accuracy"]],
    on="Classifier",
    suffixes=("_Baseline", "_Extended"),
)
summary_df["Improvement"] = summary_df["Accuracy_Extended"] - summary_df["Accuracy_Baseline"]
summary_df = summary_df.set_index("Classifier")

pd.options.display.float_format = '{:,.4f}'.format
print(summary_df)
print("\n" + "=" * 80)
print("Preparatory analysis complete.")