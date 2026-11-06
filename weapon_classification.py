"""
Classical Machine Learning for Weapon Image Classification
Author: Comus Bala
Coursework: PhD Machine Learning, Adamas University

Expected dataset structure:
dataset/
    Knife/
        image001.jpg
    Pistol/
        image001.jpg
    Rifle/
        image001.jpg
    AK-47/
        image001.jpg
    Shotgun/
        image001.jpg

Install:
    pip install -r requirements.txt

Run:
    python weapon_classification.py --data_dir dataset --output_dir results

This script uses ONLY classical machine learning (no deep learning).
It evaluates:
  E1 HOG + SVM
  E2 LBP + SVM
  E3 GLCM + SVM
  E4 HOG+LBP + SVM
  E5 HOG+LBP + Random Forest
  E6 HOG+LBP + KNN
  E7 HOG+LBP + Decision Tree
  E8 HOG+LBP + Gaussian Naive Bayes

The held-out test set is used only for final evaluation. Hyperparameter
selection uses cross-validation on the training partition.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    images, labels, paths = [], [], []
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Create one subfolder per class inside the dataset directory.")

    for class_dir in class_dirs:
        files = sorted(p for p in class_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        for file in files:
            image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: could not read {file}")
                continue
            image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            images.append(image)
            labels.append(class_dir.name)
            paths.append(str(file))

    if not images:
        raise ValueError("No readable images found. Check the class folders and image files.")

    counts = pd.Series(labels).value_counts().sort_index()
    print("\nImages per class:")
    print(counts.to_string())
    if counts.min() < 5:
        raise ValueError("Each class needs at least 5 images for a stratified train/test split and CV.")

    return np.asarray(images), np.asarray(labels), paths


def extract_hog(image):
    # 128x128, 9 orientations, 8x8 pixels/cell, 2x2 cells/block -> 8100 values
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    ).astype(np.float32)


def extract_lbp(image, points=8, radius=1):
    lbp = local_binary_pattern(image, P=points, R=radius, method="uniform")
    bins = points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, bins + 1), range=(0, bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_glcm(image):
    # Reduce gray levels to 32 to keep the co-occurrence matrix compact.
    quantized = (image // 8).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=[1, 2, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True
    )
    features = []
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        features.extend(graycoprops(matrix, prop).ravel())
    return np.asarray(features, dtype=np.float32)


def build_feature_sets(images):
    print("\nExtracting HOG, LBP and GLCM features...")
    hog_features, lbp_features, glcm_features = [], [], []
    for i, image in enumerate(images, start=1):
        hog_features.append(extract_hog(image))
        lbp_features.append(extract_lbp(image))
        glcm_features.append(extract_glcm(image))
        if i % 100 == 0:
            print(f"  processed {i}/{len(images)} images")

    H = np.vstack(hog_features)
    L = np.vstack(lbp_features)
    G = np.vstack(glcm_features)
    return {
        "HOG": H,
        "LBP": L,
        "GLCM": G,
        "HOG+LBP": np.hstack([H, L])
    }


def model_and_grid(experiment_name):
    # Pipelines prevent scaler leakage: scaling is fitted within each CV fold.
    if experiment_name.endswith("SVM"):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
        grid = [
            {
                "clf__kernel": ["linear"],
                "clf__C": [0.1, 1, 10, 100],
                "clf__class_weight": [None, "balanced"]
            },
            {
                "clf__kernel": ["rbf"],
                "clf__C": [0.1, 1, 10, 100],
                "clf__gamma": ["scale", 0.001, 0.01, 0.1],
                "clf__class_weight": [None, "balanced"]
            }
        ]
    elif experiment_name.endswith("Random Forest"):
        pipe = Pipeline([("clf", RandomForestClassifier(random_state=42, n_jobs=-1))])
        grid = {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__max_features": ["sqrt", "log2"]
        }
    elif experiment_name.endswith("KNN"):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
        grid = {
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]
        }
    elif experiment_name.endswith("Decision Tree"):
        pipe = Pipeline([("clf", DecisionTreeClassifier(random_state=42))])
        grid = {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4]
        }
    elif experiment_name.endswith("Naive Bayes"):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())])
        grid = {"clf__var_smoothing": np.logspace(-11, -7, 5)}
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    return pipe, grid


def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset", help="Folder containing one subfolder per class")
    parser.add_argument("--output_dir", default="results", help="Where to save metrics, plots and models")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "confusion_matrices").mkdir(exist_ok=True)

    images, y, paths = load_dataset(args.data_dir)
    X_sets = build_feature_sets(images)
    class_names = sorted(np.unique(y))

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=y
    )
    y_train, y_test = y[train_idx], y[test_idx]

    experiments = [
        ("E1 HOG + SVM", "HOG", "SVM"),
        ("E2 LBP + SVM", "LBP", "SVM"),
        ("E3 GLCM + SVM", "GLCM", "SVM"),
        ("E4 HOG+LBP + SVM", "HOG+LBP", "SVM"),
        ("E5 HOG+LBP + Random Forest", "HOG+LBP", "Random Forest"),
        ("E6 HOG+LBP + KNN", "HOG+LBP", "KNN"),
        ("E7 HOG+LBP + Decision Tree", "HOG+LBP", "Decision Tree"),
        ("E8 HOG+LBP + Naive Bayes", "HOG+LBP", "Naive Bayes"),
    ]

    results = []
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    for exp_name, feature_name, classifier_name in experiments:
        print(f"\n{'=' * 72}\n{exp_name}\n{'=' * 72}")
        X = X_sets[feature_name]
        X_train, X_test = X[train_idx], X[test_idx]

        pipe, param_grid = model_and_grid(f"{exp_name} {classifier_name}")
        search = GridSearchCV(
            pipe, param_grid, scoring="f1_macro", cv=cv,
            n_jobs=-1, refit=True, verbose=0, error_score="raise"
        )

        start = time.perf_counter()
        search.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = search.best_estimator_.predict(X_test)
        prediction_seconds = time.perf_counter() - start

        row = {
            "experiment": exp_name,
            "feature_method": feature_name,
            "classifier": classifier_name,
            "feature_dimensions": int(X.shape[1]),
            "cv_best_macro_f1": float(search.best_score_),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "test_recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "test_f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "training_and_search_seconds": float(training_seconds),
            "test_prediction_seconds": float(prediction_seconds),
            "best_parameters": json.dumps(search.best_params_, default=str)
        }
        results.append(row)

        print(f"Best CV macro-F1: {row['cv_best_macro_f1']:.4f}")
        print(f"Test accuracy:    {row['test_accuracy']:.4f}")
        print(f"Test macro-F1:    {row['test_f1_macro']:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, predictions, labels=class_names, zero_division=0))

        safe_name = exp_name.lower().replace(" ", "_").replace("+", "plus")
        save_confusion_matrix(
            y_test, predictions, class_names, exp_name,
            output / "confusion_matrices" / f"{safe_name}.png"
        )
        joblib.dump(search.best_estimator_, output / "models" / f"{safe_name}.joblib")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output / "experiment_results.csv", index=False)
    results_df.to_excel(output / "experiment_results.xlsx", index=False)

    # Comparative metric chart
    plot_df = results_df.set_index("experiment")[
        ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
    ]
    ax = plot_df.plot(kind="bar", figsize=(14, 7), ylim=(0, 1))
    ax.set_ylabel("Score")
    ax.set_xlabel("Experiment")
    ax.set_title("Classical ML Weapon Image Classification: Test Metrics")
    ax.legend(loc="lower right")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output / "classifier_comparison.png", dpi=220)
    plt.close()

    # Training/search time chart
    ax = results_df.set_index("experiment")["training_and_search_seconds"].plot(
        kind="bar", figsize=(12, 6)
    )
    ax.set_ylabel("Seconds")
    ax.set_title("Training and Hyperparameter Search Time")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output / "training_time_comparison.png", dpi=220)
    plt.close()

    # Feature dimension chart
    feature_dims = pd.DataFrame([
        {"feature": name, "dimensions": int(matrix.shape[1])}
        for name, matrix in X_sets.items()
    ])
    feature_dims.to_csv(output / "feature_dimensions.csv", index=False)
    ax = feature_dims.set_index("feature")["dimensions"].plot(
        kind="bar", figsize=(8, 5)
    )
    ax.set_ylabel("Number of features")
    ax.set_title("Feature Vector Dimension Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output / "feature_dimension_comparison.png", dpi=220)
    plt.close()

    print("\nCompleted.")
    print(f"Results saved in: {output.resolve()}")
    print("Key files: experiment_results.csv, experiment_results.xlsx,")
    print("classifier_comparison.png, training_time_comparison.png,")
    print("feature_dimension_comparison.png, confusion_matrices/, models/")


if __name__ == "__main__":
    main()
