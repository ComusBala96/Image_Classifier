
import os
import json
import argparse
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = 128

HOG_FEATURES = 8100
LBP_FEATURES = 10
GLCM_FEATURES = 72

TOTAL_FEATURES = HOG_FEATURES + LBP_FEATURES + GLCM_FEATURES

# Default number of training images used per experiment.
# Set --max_train_samples 0 to use the complete training set.
DEFAULT_MAX_TRAIN_SAMPLES = 8000

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

RANDOM_STATE = 42


# ============================================================
# 2. DATASET DISCOVERY
# ============================================================

def discover_classes(data_dir):

    train_dir = Path(data_dir) / "train"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    classes = sorted(
        folder.name
        for folder in train_dir.iterdir()
        if folder.is_dir()
    )

    if len(classes) < 2:
        raise ValueError(
            "At least two class folders are required."
        )

    print("\nDetected categories:", len(classes))

    for index, class_name in enumerate(classes):
        print(f"{index:03d} - {class_name}")

    return classes


def scan_split(split_dir, classes):

    split_dir = Path(split_dir)

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_dir}"
        )

    paths = []
    labels = []

    for class_name in classes:

        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing category folder: {class_dir}"
            )

        images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        for image_path in images:
            paths.append(str(image_path))
            labels.append(class_name)

    return paths, labels


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return image


# ============================================================
# 4. HOG FEATURE EXTRACTION
# ============================================================

def extract_hog_features(image):

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    return features.astype(np.float32)


# ============================================================
# 5. LBP FEATURE EXTRACTION
# ============================================================

def extract_lbp_features(image):

    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        image,
        points,
        radius,
        method="uniform"
    )

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, points + 3),
        range=(0, points + 2)
    )

    histogram = histogram.astype(np.float32)

    histogram /= (histogram.sum() + 1e-8)

    return histogram


# ============================================================
# 6. GLCM FEATURE EXTRACTION
# ============================================================

def extract_glcm_features(image):

    # Reduce grayscale levels to 32.
    quantized = (image // 8).astype(np.uint8)

    distances = [1, 2, 3]

    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "ASM",
        "energy",
        "correlation"
    ]

    features = []

    for prop in properties:

        values = graycoprops(glcm, prop)

        features.extend(values.flatten())

    return np.asarray(features, dtype=np.float32)


# ============================================================
# 7. COMBINED FEATURE EXTRACTION
# ============================================================

def extract_all_features(image):

    hog_features = extract_hog_features(image)

    lbp_features = extract_lbp_features(image)

    glcm_features = extract_glcm_features(image)

    combined = np.concatenate([
        hog_features,
        lbp_features,
        glcm_features
    ])

    return combined.astype(np.float32)


# ============================================================
# 8. FEATURE CACHE
# ============================================================

def get_cache_paths(output_dir, split):

    cache_dir = Path(output_dir) / "feature_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)

    feature_path = cache_dir / f"{split}_features.npy"
    label_path = cache_dir / f"{split}_labels.npy"
    metadata_path = cache_dir / f"{split}_metadata.json"

    return feature_path, label_path, metadata_path


def extract_split_features(
    data_dir,
    split,
    classes,
    label_encoder,
    output_dir,
    force_rebuild=False
):

    feature_path, label_path, metadata_path = get_cache_paths(
        output_dir,
        split
    )

    paths, labels = scan_split(
        Path(data_dir) / split,
        classes
    )

    if not paths:
        raise ValueError(
            f"No images found in split: {split}"
        )

    expected_metadata = {
        "split": split,
        "image_size": IMAGE_SIZE,
        "total_features": TOTAL_FEATURES,
        "classes": classes,
        "image_count": len(paths)
    }

    # Reuse a cache only when its metadata matches.
    if (
        not force_rebuild
        and feature_path.exists()
        and label_path.exists()
        and metadata_path.exists()
    ):

        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                old_metadata = json.load(file)

            if old_metadata == expected_metadata:

                cached_features = np.load(
                    feature_path,
                    mmap_mode="r"
                )

                cached_labels = np.load(
                    label_path,
                    mmap_mode="r"
                )

                if (
                    cached_features.shape == (
                        len(paths), TOTAL_FEATURES
                    )
                    and cached_labels.shape == (len(paths),)
                ):

                    print(
                        f"\nUsing cached {split} features: "
                        f"{cached_features.shape}"
                    )

                    return cached_features, cached_labels

        except Exception:
            print("Existing cache is invalid. Rebuilding...")

    print(f"\nExtracting features for: {split}")
    print("Images:", len(paths))

    encoded_labels = label_encoder.transform(labels)

    # Disk-backed NumPy arrays avoid keeping every feature
    # vector in RAM during extraction.
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(paths), TOTAL_FEATURES)
    )

    label_array = np.asarray(
        encoded_labels,
        dtype=np.int32
    )

    failed_images = 0

    for index, image_path in enumerate(
        tqdm(paths, desc=f"{split} feature extraction")
    ):

        image = preprocess_image(image_path)

        if image is None:

            # Preserve array alignment if an image is unreadable.
            features[index] = np.zeros(
                TOTAL_FEATURES,
                dtype=np.float32
            )

            failed_images += 1
            continue

        features[index] = extract_all_features(image)

    features.flush()

    np.save(label_path, label_array)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            expected_metadata,
            file,
            indent=4
        )

    print(f"Failed images: {failed_images}")

    return (
        np.load(feature_path, mmap_mode="r"),
        np.load(label_path, mmap_mode="r")
    )


# ============================================================
# 9. FEATURE INDEX DEFINITIONS
# ============================================================

HOG_SLICE = slice(
    0,
    HOG_FEATURES
)

LBP_SLICE = slice(
    HOG_FEATURES,
    HOG_FEATURES + LBP_FEATURES
)

GLCM_SLICE = slice(
    HOG_FEATURES + LBP_FEATURES,
    TOTAL_FEATURES
)

HOG_LBP_SLICE = slice(
    0,
    HOG_FEATURES + LBP_FEATURES
)

LBP_GLCM_SLICE = slice(
    HOG_FEATURES,
    TOTAL_FEATURES
)


# ============================================================
# 10. MEMORY-CONSCIOUS TRAINING SAMPLE
# ============================================================

def select_training_indices(
    labels,
    max_train_samples,
    random_state=RANDOM_STATE
):

    labels = np.asarray(labels)

    if max_train_samples <= 0 or len(labels) <= max_train_samples:
        return np.arange(len(labels))

    print(
        f"\nSampling {max_train_samples:,} images "
        f"from {len(labels):,} training images."
    )

    indices = np.arange(len(labels))

    selected, _ = train_test_split(
        indices,
        train_size=max_train_samples,
        stratify=labels,
        random_state=random_state
    )

    return np.sort(selected)


# ============================================================
# 11. CLASSICAL MACHINE-LEARNING MODELS
# ============================================================

def build_experiments():

    experiments = {

        "01_HOG_LinearSVM": {
            "features": HOG_SLICE,
            "model": Pipeline([
                (
                    "scaler",
                    StandardScaler(with_mean=False)
                ),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE
                    )
                )
            ])
        },

        "02_LBP_LinearSVM": {
            "features": LBP_SLICE,
            "model": Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE
                    )
                )
            ])
        },

        "03_GLCM_LinearSVM": {
            "features": GLCM_SLICE,
            "model": Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE
                    )
                )
            ])
        },

        "04_HOG_LBP_LinearSVM": {
            "features": HOG_LBP_SLICE,
            "model": Pipeline([
                (
                    "scaler",
                    StandardScaler(with_mean=False)
                ),
                (
                    "classifier",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE
                    )
                )
            ])
        },

        "05_HOG_LBP_RandomForest": {
            "features": HOG_LBP_SLICE,
            "model": RandomForestClassifier(
                n_estimators=150,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        },

        # KNN uses the compact LBP + GLCM representation.
        # This avoids extremely expensive distance calculations
        # over thousands of HOG dimensions.
        "06_LBP_GLCM_KNN": {
            "features": LBP_GLCM_SLICE,
            "model": Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                        n_jobs=-1
                    )
                )
            ])
        },

        "07_HOG_LBP_DecisionTree": {
            "features": HOG_LBP_SLICE,
            "model": DecisionTreeClassifier(
                criterion="gini",
                max_depth=40,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        },

        # GaussianNB also uses compact texture features
        # to reduce memory and computational requirements.
        "08_LBP_GLCM_GaussianNB": {
            "features": LBP_GLCM_SLICE,
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB())
            ])
        }
    }

    return experiments


# ============================================================
# 12. BATCH PREDICTION
# ============================================================

def predict_in_batches(
    model,
    features,
    feature_slice,
    batch_size=512
):

    predictions = []

    for start in range(0, len(features), batch_size):

        end = min(
            start + batch_size,
            len(features)
        )

        batch = np.asarray(
            features[start:end, feature_slice],
            dtype=np.float32
        )

        batch_predictions = model.predict(batch)

        predictions.extend(batch_predictions)

    return np.asarray(predictions)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    features,
    labels,
    feature_slice,
    class_names,
    split_name
):

    print(f"\nEvaluating on {split_name} data...")

    predictions = predict_in_batches(
        model,
        features,
        feature_slice
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions
    )

    macro_precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    results = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    }

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    report = classification_report(
        labels,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    return results, predictions, report_df


# ============================================================
# 14. SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    labels,
    predictions,
    class_names,
    output_dir
):

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(len(class_names))
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    )

    matrix_df.to_csv(
        Path(output_dir) / "confusion_matrix.csv"
    )

    # A large class count makes a plotted matrix difficult
    # to read. Save a figure with a compact layout as well.
    fig_size = max(12, len(class_names) * 0.18)

    fig, ax = plt.subplots(
        figsize=(fig_size, fig_size)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto"
    )

    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    if len(class_names) <= 60:

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))

        ax.set_xticklabels(
            class_names,
            rotation=90,
            fontsize=6
        )

        ax.set_yticklabels(
            class_names,
            fontsize=6
        )

    else:

        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(image, ax=ax)

    plt.tight_layout()

    plt.savefig(
        Path(output_dir) / "confusion_matrix.png",
        dpi=200
    )

    plt.close(fig)


# ============================================================
# 15. PLOT MODEL COMPARISON
# ============================================================

def save_comparison_chart(results_df, output_dir):

    if results_df.empty:
        return

    plot_df = results_df.sort_values(
        "Macro F1",
        ascending=False
    )

    plt.figure(figsize=(13, 7))

    plt.bar(
        plot_df["Experiment"],
        plot_df["Macro F1"]
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylabel("Macro F1-score")
    plt.xlabel("Machine Learning Experiment")
    plt.title("Classical Machine Learning Model Comparison")

    plt.tight_layout()

    plt.savefig(
        Path(output_dir) / "model_comparison.png",
        dpi=200
    )

    plt.close()


# ============================================================
# 16. MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Classical Multi-Class Image Classification"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Dataset root directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory"
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=DEFAULT_MAX_TRAIN_SAMPLES,
        help="Maximum training images per experiment; 0 means all"
    )

    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Rebuild cached image features"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("CLASSICAL MACHINE LEARNING IMAGE CLASSIFICATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Discover all classes
    # --------------------------------------------------------

    classes = discover_classes(data_dir)

    label_encoder = LabelEncoder()

    label_encoder.fit(classes)

    with open(
        output_dir / "class_mapping.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                str(index): class_name
                for index, class_name in enumerate(
                    label_encoder.classes_
                )
            },
            file,
            indent=4
        )

    print("\nTotal categories:", len(classes))

    # --------------------------------------------------------
    # Extract or load features
    # --------------------------------------------------------

    X_train, y_train = extract_split_features(
        data_dir,
        "train",
        classes,
        label_encoder,
        output_dir,
        args.force_rebuild
    )

    X_valid, y_valid = extract_split_features(
        data_dir,
        "valid",
        classes,
        label_encoder,
        output_dir,
        args.force_rebuild
    )

    X_test, y_test = extract_split_features(
        data_dir,
        "test",
        classes,
        label_encoder,
        output_dir,
        args.force_rebuild
    )

    print("\nDataset summary")
    print("-" * 50)
    print("Training images:", len(y_train))
    print("Validation images:", len(y_valid))
    print("Testing images:", len(y_test))
    print("Number of classes:", len(classes))
    print("Features per image:", TOTAL_FEATURES)

    # --------------------------------------------------------
    # Select a stratified training sample
    # --------------------------------------------------------

    train_indices = select_training_indices(
        y_train,
        args.max_train_samples
    )

    y_train_selected = np.asarray(
        y_train[train_indices]
    )

    # --------------------------------------------------------
    # Prepare experiments
    # --------------------------------------------------------

    experiments = build_experiments()

    comparison_results = []

    best_macro_f1 = -1
    best_model = None
    best_experiment = None
    best_feature_slice = None

    model_dir = output_dir / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Train and validate all experiments
    # --------------------------------------------------------

    for experiment_name, experiment in experiments.items():

        print("\n" + "=" * 65)
        print("EXPERIMENT:", experiment_name)
        print("=" * 65)

        feature_slice = experiment["features"]

        model = experiment["model"]

        print("Preparing training features...")

        X_train_selected = np.asarray(
            X_train[train_indices, feature_slice],
            dtype=np.float32
        )

        print(
            "Training feature shape:",
            X_train_selected.shape
        )

        print("Training model...")

        model.fit(
            X_train_selected,
            y_train_selected
        )

        print("Validation evaluation...")

        validation_results, _, validation_report = evaluate_model(
            model,
            X_valid,
            y_valid,
            feature_slice,
            classes,
            "Validation"
        )

        validation_report.to_csv(
            output_dir / f"{experiment_name}_validation_report.csv"
        )

        row = {
            "Experiment": experiment_name,
            **validation_results
        }

        comparison_results.append(row)

        # Save the trained experiment.
        joblib.dump(
            {
                "model": model,
                "feature_slice": (
                    feature_slice.start,
                    feature_slice.stop
                ),
                "experiment": experiment_name,
                "classes": classes,
                "image_size": IMAGE_SIZE
            },
            model_dir / f"{experiment_name}.joblib"
        )

        if validation_results["Macro F1"] > best_macro_f1:

            best_macro_f1 = validation_results["Macro F1"]

            best_model = model

            best_experiment = experiment_name

            best_feature_slice = feature_slice

            print("\nNew best model:", best_experiment)

        del X_train_selected

    # --------------------------------------------------------
    # Save comparison results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        comparison_results
    )

    results_df = results_df.sort_values(
        "Macro F1",
        ascending=False
    )

    results_df.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    save_comparison_chart(
        results_df,
        output_dir
    )

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df.to_string(index=False)
    )

    print("\nBest validation model:", best_experiment)
    print(f"Best validation Macro F1: {best_macro_f1:.4f}")

    # --------------------------------------------------------
    # Save best model separately
    # --------------------------------------------------------

    best_model_path = model_dir / "best_model.joblib"

    joblib.dump(
        {
            "model": best_model,
            "feature_slice": (
                best_feature_slice.start,
                best_feature_slice.stop
            ),
            "experiment": best_experiment,
            "classes": classes,
            "image_size": IMAGE_SIZE
        },
        best_model_path
    )

    # --------------------------------------------------------
    # Final test evaluation
    # Test set is used only for the selected model.
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION")
    print("=" * 65)

    test_results, test_predictions, test_report = evaluate_model(
        best_model,
        X_test,
        y_test,
        best_feature_slice,
        classes,
        "Test"
    )

    test_report.to_csv(
        output_dir / "best_model_test_report.csv"
    )

    with open(
        output_dir / "final_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            test_results,
            file,
            indent=4
        )

    save_confusion_matrix(
        y_test,
        test_predictions,
        classes,
        output_dir
    )

    # --------------------------------------------------------
    # Save experiment summary
    # --------------------------------------------------------

    summary = {
        "project": "Classical Multi-Class Image Classification",
        "number_of_classes": len(classes),
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "hog_features": HOG_FEATURES,
        "lbp_features": LBP_FEATURES,
        "glcm_features": GLCM_FEATURES,
        "total_features": TOTAL_FEATURES,
        "training_images": len(y_train),
        "validation_images": len(y_valid),
        "testing_images": len(y_test),
        "training_samples_per_experiment": len(train_indices),
        "best_experiment": best_experiment,
        "best_validation_macro_f1": best_macro_f1,
        "test_metrics": test_results
    }

    with open(
        output_dir / "experiment_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("CLASSIFICATION COMPLETED")
    print("=" * 65)

    print("Best model:", best_experiment)
    print("Model saved:", best_model_path)
    print("Output directory:", output_dir.resolve())

    print("\nGenerated files:")
    print("1. model_comparison.csv")
    print("2. model_comparison.png")
    print("3. best_model_test_report.csv")
    print("4. final_test_metrics.json")
    print("5. confusion_matrix.csv")
    print("6. confusion_matrix.png")
    print("7. experiment_summary.json")
    print("8. class_mapping.json")
    print("9. models/")
    print("10. feature_cache/")


# ============================================================
# 17. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
