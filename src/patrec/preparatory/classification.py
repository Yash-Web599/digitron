"""
This module handles the classification tasks for the preparatory analysis,
including dimensionality reduction with PCA and systematic classifier tuning
using GridSearchCV.
"""
import time
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def apply_pca(
    X_train: np.ndarray, X_test: np.ndarray, n_components: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Applies PCA to the training and test data.

    The PCA model is fitted ONLY on the training data and then used to
    transform both the training and test data. This prevents data leakage.

    Args:
        X_train (np.ndarray): The training feature set.
        X_test (np.ndarray): The test feature set.
        n_components (float): The amount of variance to preserve. For example,
                              0.95 means PCA will select the number of components
                              that preserve 95% of the variance.

    Returns:
        Tuple[np.ndarray, np.ndarray, PCA]: A tuple containing the transformed
                                            training data, transformed test data,
                                            and the fitted PCA object.
    """
    # Initialize PCA to retain a certain percentage of variance.
    # This is often more robust than picking a fixed number of components.
    pca = PCA(n_components=n_components, random_state=42)

    # Fit PCA on the training data and transform it.
    X_train_pca = pca.fit_transform(X_train)

    # Apply the SAME transformation to the test data.
    X_test_pca = pca.transform(X_test)

    print(f"PCA applied. Dimensions reduced to {pca.n_components_} to preserve {n_components:.0%} of variance.")
    return X_train_pca, X_test_pca, pca


def get_classifiers_and_grids() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Defines the classifiers and their corresponding hyperparameter grids for tuning.

    Returns:
        Dict[str, Tuple[Any, Dict[str, Any]]]: A dictionary where keys are classifier
                                               names and values are tuples containing
                                               the classifier instance and its param grid.
    """
    # Define the machine learning models and the hyperparameters to search through.
    # This dictionary makes it easy to add or remove classifiers from the experiment.
    classifiers = {
        "Naive Bayes": (GaussianNB(), {"var_smoothing": np.logspace(-2, -10, num=12)}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5],
            },
        ),
        "SVM": (
            SVC(random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        ),
        "k-NN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=4000, random_state=42),
            {"C": [0.01, 0.1, 1, 10], "solver": ["saga"]},
        ),
    }
    return classifiers


def run_classification_pipeline(
    X: np.ndarray, y: List[int], feature_set_name: str
) -> List[Dict[str, Any]]:
    """
    Executes the full classification pipeline for a given feature set.

    This function encapsulates the standard machine learning workflow: data splitting,
    scaling, hyperparameter tuning via cross-validation, and final evaluation.

    Args:
        X (np.ndarray): The full feature matrix (e.g., utterance vectors).
        y (List[int]): The list of corresponding labels for each row in X.
        feature_set_name (str): A descriptive name for the feature set being used
                                (e.g., "Baseline", "Extended"). This is used for
                                logging and organizing results.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries. Each dictionary contains the
                              detailed evaluation results for one classifier, including
                              its name, accuracy, best parameters, and confusion matrix.
    """
    print("\n" + "=" * 80)
    print(f"STARTING CLASSIFICATION PIPELINE FOR: {feature_set_name.upper()} FEATURES")
    print("=" * 80)

    # 1. Split data into training and testing sets.
    # 'random_state' ensures the split is the same every time, for reproducibility.
    # 'stratify=y' is crucial: it ensures the proportion of each digit is identical
    # in both the training and test sets, preventing skewed evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2. Scale the features using StandardScaler.
    # The scaler is fitted ONLY on the training data. We then use the
    # fitted scaler to transform both the training and test data. This prevents
    # any information ("data leakage") from the test set influencing the model.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Get the classifiers and their hyperparameter grids for tuning.
    classifiers = get_classifiers_and_grids()
    results = []

    # 4. Iterate through each classifier to perform tuning and evaluation.
    for name, (estimator, param_grid) in classifiers.items():
        print(f"\n--- Tuning {name} ---")
        start_time = time.time()

        # GridSearchCV performs an exhaustive search over the specified parameter grid.
        # 'cv=5' specifies 5-fold cross-validation.
        # 'n_jobs=-1' utilizes all available CPU cores, significantly speeding up the search.
        grid_search = GridSearchCV(
            estimator, param_grid, cv=5, n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)

        # 5. Evaluate the best model found by GridSearchCV on the unseen test set.
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 6. Compute the confusion matrix for detailed error analysis.
        # Providing 'labels' ensures the matrix has a consistent size and order.
        cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y)))

        duration = time.time() - start_time
        print(f"-> Best parameters: {grid_search.best_params_}")
        print(f"-> Test Accuracy: {accuracy:.4f} (Tuning took {duration:.2f}s)")

        # 7. Store all relevant results in a dictionary.
        results.append(
            {
                "Feature Set": feature_set_name,
                "Classifier": name,
                "Accuracy": accuracy,
                "Best Params": grid_search.best_params_,
                "confusion_matrix": cm, 
            }
        )
    return results