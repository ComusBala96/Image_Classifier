
import argparse
import gc
import json
import warnings
from pathlib import Path

import cv2
import cupy as cp
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 128

HOG_FEATURES = 8100
LBP_FEATURES = 10
GLCM_FEATURES = 72
TOTAL_FEATURES = HOG_FEATURES + LBP_FEATURES + GLCM_FEATURES

DEFAULT_MAX_TRAIN_SAMPLES = 8000
RANDOM_STATE = 42

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GPU_BATCH_SIZE = 512
SVM_EPOCHS = 30
SVM_LEARNING_RATE = 0.01

# ============================================================
# GPU INITIALIZATION
# ============================================================

def initialize_gpu():

    print("\nGPU CONFIGURATION")
    print("=" * 55)

    print("PyTorch device:", DEVICE)

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA is not available in PyTorch. "
            "Install a CUDA-enabled PyTorch build first."
        )

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy cannot detect an NVIDIA GPU.")

    properties = cp.cuda.runtime.getDeviceProperties(0)

    gpu_name = properties["name"]

    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    print("CuPy GPU:", gpu_name)
    print("PyTorch GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

    total_memory = torch.cuda.get_device_properties(0).total_memory

    print(
        "GPU memory:",
        round(total_memory / (1024 ** 3), 2),
        "GB"
    )

    cp.cuda.Device(0).use()


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_classes(data_dir):

    train_dir = Path(data_dir) / "train"

    if not train_dir.is_dir():
        raise FileNotFoundError(train_dir)

    classes = sorted(
        folder.name
        for folder in train_dir.iterdir()
        if folder.is_dir()
    )

    if len(classes) < 2:
        raise ValueError("At least two classes are required.")

    print("\nDetected classes:", len(classes))

    for index, name in enumerate(classes):
        print(f"{index:03d}: {name}")

    return classes


def scan_split(split_dir, classes):

    paths = []
    labels = []

    split_dir = Path(split_dir)

    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)

    for class_name in classes:

        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for path in sorted(class_dir.rglob("*")):

            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                paths.append(str(path))
                labels.append(class_name)

    return paths, labels


# ============================================================
# GPU HOG FEATURE EXTRACTION
# ============================================================

def gpu_hog(image):

    img = cp.asarray(image, dtype=cp.float32) / 255.0

    # Gradients
    gx = cp.zeros_like(img)
    gy = cp.zeros_like(img)

    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    magnitude = cp.sqrt(gx * gx + gy * gy)

    angle = cp.arctan2(gy, gx) * (180.0 / cp.pi)
    angle = cp.mod(angle, 180.0)

    orientations = 9

    bin_index = cp.floor(
        angle * orientations / 180.0
    ).astype(cp.int32)

    bin_index = cp.clip(
        bin_index,
        0,
        orientations - 1
    )

    # 128 x 128 image, 8 x 8 pixel cells.
    cell_size = 8
    cells_per_axis = IMAGE_SIZE // cell_size

    rows = cp.arange(IMAGE_SIZE)[:, None]
    cols = cp.arange(IMAGE_SIZE)[None, :]

    cell_y = rows // cell_size
    cell_x = cols // cell_size

    cell_index = (
        (cell_y * cells_per_axis + cell_x) * orientations
        + bin_index
    )

    histogram = cp.bincount(
        cell_index.ravel(),
        weights=magnitude.ravel(),
        minlength=(
            cells_per_axis * cells_per_axis * orientations
        )
    )

    histogram = histogram.reshape(
        cells_per_axis,
        cells_per_axis,
        orientations
    )

    # 2 x 2 cell blocks, 15 x 15 blocks.
    blocks = []

    for y in range(cells_per_axis - 1):
        for x in range(cells_per_axis - 1):

            block = histogram[
                y:y + 2,
                x:x + 2,
                :
            ].ravel()

            norm = cp.sqrt(
                cp.sum(block * block) + 1e-8
            )

            block = block / norm

            block = cp.minimum(block, 0.2)

            block = block / (
                cp.sqrt(cp.sum(block * block) + 1e-8)
            )

            blocks.append(block)

    return cp.concatenate(blocks).astype(cp.float32)


# ============================================================
# GPU LBP FEATURE EXTRACTION
# ============================================================

def gpu_lbp(image):

    img = cp.asarray(image, dtype=cp.uint8)

    center = img.astype(cp.int16)

    lbp = cp.zeros(
        img.shape,
        dtype=cp.uint16
    )

    # Eight circular neighbors.
    shifts = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1)
    ]

    for bit, (dy, dx) in enumerate(shifts):

        neighbor = cp.roll(
            img,
            shift=(dy, dx),
            axis=(0, 1)
        ).astype(cp.int16)

        lbp |= (
            (neighbor >= center).astype(cp.uint16) << bit
        )

    # Uniform LBP: 8 neighbors + one non-uniform bin.
    # 10 bins in total.
    transitions = cp.zeros_like(lbp)

    for bit in range(8):

        current = (lbp >> bit) & 1
        following = (lbp >> ((bit + 1) % 8)) & 1

        transitions += (current != following).astype(cp.uint16)

    uniform = transitions <= 2

    # Number of set bits for uniform patterns.
    popcount = cp.zeros_like(lbp)

    for bit in range(8):
        popcount += ((lbp >> bit) & 1).astype(cp.uint16)

    bins = cp.where(
        uniform,
        popcount,
        9
    ).astype(cp.int32)

    histogram = cp.bincount(
        bins.ravel(),
        minlength=10
    ).astype(cp.float32)

    histogram /= cp.maximum(
        histogram.sum(),
        1.0
    )

    return histogram


# ============================================================
# GPU GLCM FEATURE EXTRACTION
# ============================================================

def gpu_glcm(image):

    quantized = (
        cp.asarray(image, dtype=cp.uint8) // 8
    ).astype(cp.int32)

    levels = 32

    offsets = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1)
    ]

    distances = [1, 2, 3]

    features = []

    for distance in distances:

        for dy, dx in offsets:

            dy *= distance
            dx *= distance

            y1 = max(0, -dy)
            y2 = min(IMAGE_SIZE, IMAGE_SIZE - dy)

            x1 = max(0, -dx)
            x2 = min(IMAGE_SIZE, IMAGE_SIZE - dx)

            first = quantized[y1:y2, x1:x2]
            second = quantized[y1 + dy:y2 + dy, x1 + dx:x2 + dx]

            encoded = first.ravel() * levels + second.ravel()

            counts = cp.bincount(
                encoded,
                minlength=levels * levels
            ).reshape(levels, levels).astype(cp.float32)

            # Symmetric GLCM
            matrix = counts + counts.T

            matrix /= cp.maximum(
                matrix.sum(),
                1.0
            )

            i = cp.arange(levels, dtype=cp.float32)[:, None]
            j = cp.arange(levels, dtype=cp.float32)[None, :]

            difference = cp.abs(i - j)

            contrast = cp.sum(
                matrix * (difference ** 2)
            )

            dissimilarity = cp.sum(
                matrix * difference
            )

            homogeneity = cp.sum(
                matrix / (1.0 + difference ** 2)
            )

            asm = cp.sum(matrix ** 2)

            energy = cp.sqrt(asm)

            mean_i = cp.sum(
                matrix * i
            )

            mean_j = cp.sum(
                matrix * j
            )

            std_i = cp.sqrt(
                cp.sum(matrix * (i - mean_i) ** 2)
            )

            std_j = cp.sqrt(
                cp.sum(matrix * (j - mean_j) ** 2)
            )

            covariance = cp.sum(
                matrix * (i - mean_i) * (j - mean_j)
            )

            correlation = covariance / (
                std_i * std_j + 1e-8
            )

            features.extend([
                contrast,
                dissimilarity,
                homogeneity,
                asm,
                energy,
                correlation
            ])

    return cp.asarray(
        features,
        dtype=cp.float32
    )


# ============================================================
# COMBINED GPU FEATURES
# ============================================================

def extract_gpu_features(image):

    hog_features = gpu_hog(image)
    lbp_features = gpu_lbp(image)
    glcm_features = gpu_glcm(image)

    combined = cp.concatenate([
        hog_features,
        lbp_features,
        glcm_features
    ])

    return combined


# ============================================================
# FEATURE CACHE
# ============================================================

def extract_split_features(
    data_dir,
    split,
    classes,
    encoder,
    output_dir,
    force_rebuild=False
):

    cache_dir = Path(output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    feature_file = cache_dir / f"{split}_features.npy"
    label_file = cache_dir / f"{split}_labels.npy"
    metadata_file = cache_dir / f"{split}_metadata.json"

    paths, labels = scan_split(
        Path(data_dir) / split,
        classes
    )

    if not paths:
        raise ValueError(f"No images found in {split}")

    metadata = {
        "split": split,
        "classes": classes,
        "image_count": len(paths),
        "image_size": IMAGE_SIZE,
        "feature_count": TOTAL_FEATURES,
        "extractor": "cupy_hog_lbp_glcm_v1"
    }

    if (
        not force_rebuild
        and feature_file.exists()
        and label_file.exists()
        and metadata_file.exists()
    ):

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                old_metadata = json.load(f)

            if old_metadata == metadata:

                X = np.load(feature_file, mmap_mode="r")
                y = np.load(label_file, mmap_mode="r")

                if (
                    X.shape == (len(paths), TOTAL_FEATURES)
                    and y.shape == (len(paths),)
                ):
                    print(
                        f"\nUsing cached {split} features: {X.shape}"
                    )
                    return X, y

        except Exception:
            print("Cache invalid. Rebuilding features.")

    print(f"\nGPU feature extraction: {split}")
    print("Images:", len(paths))

    X = np.lib.format.open_memmap(
        feature_file,
        mode="w+",
        dtype=np.float32,
        shape=(len(paths), TOTAL_FEATURES)
    )

    y = encoder.transform(labels).astype(np.int32)

    failed = 0

    for index, image_path in enumerate(
        tqdm(paths, desc=f"GPU {split}")
    ):

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            X[index] = 0
            failed += 1
            continue

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        try:
            features = extract_gpu_features(image)

            X[index] = cp.asnumpy(features)

        except Exception as error:

            print(
                f"\nFeature error: {image_path}: {error}"
            )

            X[index] = 0
            failed += 1

    X.flush()

    np.save(label_file, y)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    cp.get_default_memory_pool().free_all_blocks()

    print("Failed images:", failed)

    return (
        np.load(feature_file, mmap_mode="r"),
        np.load(label_file, mmap_mode="r")
    )


# ============================================================
# FEATURE SLICES
# ============================================================

HOG_SLICE = slice(0, HOG_FEATURES)

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
# TRAINING SAMPLE SELECTION
# ============================================================

def select_training_indices(labels, maximum):

    labels = np.asarray(labels)

    if maximum <= 0 or len(labels) <= maximum:
        return np.arange(len(labels))

    rng = np.random.default_rng(RANDOM_STATE)

    classes, counts = np.unique(
        labels,
        return_counts=True
    )

    # Preserve at least one training example per class.
    if len(classes) > maximum:
        raise ValueError(
            "max_train_samples must be at least the number "
            "of classes."
        )

    allocation = np.ones(
        len(classes),
        dtype=np.int64
    )

    remaining = maximum - len(classes)

    if remaining > 0:

        proportions = counts / counts.sum()

        additional = np.floor(
            proportions * remaining
        ).astype(np.int64)

        allocation += np.minimum(
            additional,
            counts - allocation
        )

    # Distribute any remaining capacity.
    while allocation.sum() < maximum:

        available = np.where(allocation < counts)[0]

        if len(available) == 0:
            break

        allocation[available] += 1

    selected = []

    for class_id, amount in zip(classes, allocation):

        class_indices = np.where(labels == class_id)[0]

        chosen = rng.choice(
            class_indices,
            size=int(amount),
            replace=False
        )

        selected.extend(chosen.tolist())

    return np.sort(np.asarray(selected, dtype=np.int64))


# ============================================================
# GPU FEATURE STANDARDIZATION
# ============================================================

def gpu_scale_fit(X):

    X = torch.as_tensor(
        np.asarray(X, dtype=np.float32),
        device=DEVICE
    )

    mean = X.mean(dim=0)

    std = X.std(
        dim=0,
        unbiased=False
    )

    std = torch.where(
        std < 1e-6,
        torch.ones_like(std),
        std
    )

    return mean, std


def gpu_scale_transform(X, mean, std):

    X = torch.as_tensor(
        np.asarray(X, dtype=np.float32),
        device=DEVICE
    )

    return (X - mean) / std


# ============================================================
# GPU LINEAR SVM - ONE VS REST
# ============================================================

class GPULinearSVM:

    def __init__(
        self,
        epochs=SVM_EPOCHS,
        learning_rate=SVM_LEARNING_RATE,
        regularization=0.0005
    ):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

    def fit(self, X, y):

        X = torch.as_tensor(
            np.asarray(X, dtype=np.float32),
            device=DEVICE
        )

        y = torch.as_tensor(
            np.asarray(y, dtype=np.int64),
            device=DEVICE
        )

        self.classes_ = torch.unique(y)

        number_classes = int(self.classes_.numel())
        number_features = X.shape[1]

        self.mean_, self.std_ = gpu_scale_fit(
            X.detach().cpu().numpy()
        )

        X = (X - self.mean_) / self.std_

        # One-vs-rest target matrix.
        targets = torch.full(
            (len(y), number_classes),
            -1.0,
            device=DEVICE
        )

        targets[
            torch.arange(len(y), device=DEVICE),
            y
        ] = 1.0

        self.weights = nn.Parameter(
            torch.zeros(
                number_features,
                number_classes,
                device=DEVICE
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(
                number_classes,
                device=DEVICE
            )
        )

        optimizer = torch.optim.Adam(
            [self.weights, self.bias],
            lr=self.learning_rate
        )

        for epoch in range(self.epochs):

            optimizer.zero_grad()

            scores = X @ self.weights + self.bias

            hinge = torch.relu(
                1.0 - targets * scores
            )

            loss = hinge.mean() + (
                self.regularization
                * (self.weights ** 2).sum()
                / 2
            )

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(
                    f"GPU SVM epoch {epoch + 1}/{self.epochs} "
                    f"loss={loss.item():.4f}"
                )

        self.weights = self.weights.detach()
        self.bias = self.bias.detach()

        return self

    @torch.no_grad()
    def predict(self, X):

        X = torch.as_tensor(
            np.asarray(X, dtype=np.float32),
            device=DEVICE
        )

        X = (X - self.mean_) / self.std_

        scores = X @ self.weights + self.bias

        return scores.argmax(dim=1).cpu().numpy()


# ============================================================
# GPU KNN
# ============================================================

class GPUKNN:

    def __init__(self, k=5, batch_size=128):
        self.k = k
        self.batch_size = batch_size

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float32)

        self.mean, self.std = gpu_scale_fit(X)

        X_tensor = torch.as_tensor(
            X,
            device=DEVICE
        )

        self.X = (
            X_tensor - self.mean
        ) / self.std

        self.y = torch.as_tensor(
            np.asarray(y, dtype=np.int64),
            device=DEVICE
        )

        return self

    @torch.no_grad()
    def predict(self, X):

        predictions = []

        for start in range(0, len(X), self.batch_size):

            batch = torch.as_tensor(
                np.asarray(
                    X[start:start + self.batch_size],
                    dtype=np.float32
                ),
                device=DEVICE
            )

            batch = (batch - self.mean) / self.std

            distances = torch.cdist(
                batch,
                self.X
            )

            nearest = torch.topk(
                distances,
                k=min(self.k, len(self.y)),
                largest=False,
                dim=1
            ).indices

            nearest_labels = self.y[nearest]

            for row in nearest_labels:

                values, counts = torch.unique(
                    row,
                    return_counts=True
                )

                predictions.append(
                    values[counts.argmax()].item()
                )

            del distances

        return np.asarray(predictions)


# ============================================================
# GPU GAUSSIAN NAIVE BAYES
# ============================================================

class GPUGaussianNB:

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def fit(self, X, y):

        X = torch.as_tensor(
            np.asarray(X, dtype=np.float32),
            device=DEVICE
        )

        y = torch.as_tensor(
            np.asarray(y, dtype=np.int64),
            device=DEVICE
        )

        self.classes = torch.unique(y)

        means = []
        variances = []
        priors = []

        for class_id in self.classes:

            subset = X[y == class_id]

            means.append(
                subset.mean(dim=0)
            )

            variances.append(
                subset.var(dim=0, unbiased=False)
                + self.epsilon
            )

            priors.append(
                len(subset) / len(X)
            )

        self.means = torch.stack(means)
        self.variances = torch.stack(variances)

        self.log_priors = torch.log(
            torch.as_tensor(
                priors,
                dtype=torch.float32,
                device=DEVICE
            )
        )

        return self

    @torch.no_grad()
    def predict(self, X, batch_size=512):

        predictions = []

        for start in range(0, len(X), batch_size):

            batch = torch.as_tensor(
                np.asarray(
                    X[start:start + batch_size],
                    dtype=np.float32
                ),
                device=DEVICE
            )

            # batch: samples x features
            # class: classes x features
            difference = (
                batch[:, None, :] - self.means[None, :, :]
            )

            log_probability = -0.5 * torch.sum(
                torch.log(
                    2 * torch.pi * self.variances
                )[None, :, :]
                + (
                    difference ** 2
                    / self.variances[None, :, :]
                ),
                dim=2
            )

            scores = log_probability + self.log_priors

            predictions.extend(
                scores.argmax(dim=1).cpu().numpy()
            )

        return np.asarray(predictions)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

def build_experiments():

    return {
        "01_HOG_GPU_LinearSVM": {
            "features": HOG_SLICE,
            "model": "gpu_svm"
        },

        "02_LBP_GPU_LinearSVM": {
            "features": LBP_SLICE,
            "model": "gpu_svm"
        },

        "03_GLCM_GPU_LinearSVM": {
            "features": GLCM_SLICE,
            "model": "gpu_svm"
        },

        "04_HOG_LBP_GPU_LinearSVM": {
            "features": HOG_LBP_SLICE,
            "model": "gpu_svm"
        },

        "05_HOG_LBP_RandomForest_CPU": {
            "features": HOG_LBP_SLICE,
            "model": "random_forest"
        },

        "06_LBP_GLCM_GPU_KNN": {
            "features": LBP_GLCM_SLICE,
            "model": "gpu_knn"
        },

        "07_HOG_LBP_DecisionTree_CPU": {
            "features": HOG_LBP_SLICE,
            "model": "decision_tree"
        },

        "08_LBP_GLCM_GPU_GaussianNB": {
            "features": LBP_GLCM_SLICE,
            "model": "gpu_gnb"
        }
    }


def create_model(model_type):

    if model_type == "gpu_svm":
        return GPULinearSVM()

    if model_type == "gpu_knn":
        return GPUKNN(k=5)

    if model_type == "gpu_gnb":
        return GPUGaussianNB()

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=150,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=40,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    raise ValueError(model_type)


# ============================================================
# PREDICTION
# ============================================================

def predict_model(
    model,
    X,
    model_type,
    feature_slice
):

    if model_type.startswith("gpu_"):

        if model_type == "gpu_svm":
            return model.predict(X[:, feature_slice])

        if model_type == "gpu_knn":
            return model.predict(X[:, feature_slice])

        if model_type == "gpu_gnb":
            return model.predict(X[:, feature_slice])

    predictions = []

    for start in range(0, len(X), GPU_BATCH_SIZE):

        end = min(
            start + GPU_BATCH_SIZE,
            len(X)
        )

        batch = np.asarray(
            X[start:end, feature_slice],
            dtype=np.float32
        )

        predictions.extend(
            model.predict(batch)
        )

    return np.asarray(predictions)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    X,
    y,
    model_type,
    feature_slice,
    class_names,
    split
):

    print(f"\nEvaluating {split} set...")

    predictions = predict_model(
        model,
        X,
        model_type,
        feature_slice
    )

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y,
            predictions
        ),
        "Macro Precision": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Macro Recall": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Macro F1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Weighted F1": f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    report = classification_report(
        y,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return (
        metrics,
        predictions,
        pd.DataFrame(report).transpose()
    )


# ============================================================
# CHARTS AND REPORTS
# ============================================================

def save_comparison(results, output_dir):

    df = pd.DataFrame(results).sort_values(
        "Macro F1",
        ascending=False
    )

    df.to_csv(
        Path(output_dir) / "model_comparison.csv",
        index=False
    )

    plt.figure(figsize=(13, 7))

    plt.bar(
        df["Experiment"],
        df["Macro F1"]
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylabel("Macro F1")
    plt.title("GPU / CPU Classical ML Comparison")

    plt.tight_layout()

    plt.savefig(
        Path(output_dir) / "model_comparison.png",
        dpi=200
    )

    plt.close()

    return df


def save_confusion_matrix(y, predictions, classes, output_dir):

    matrix = confusion_matrix(
        y,
        predictions,
        labels=np.arange(len(classes))
    )

    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    ).to_csv(
        Path(output_dir) / "confusion_matrix.csv"
    )

    size = max(12, len(classes) * 0.18)

    fig, ax = plt.subplots(
        figsize=(size, size)
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    if len(classes) <= 60:

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))

        ax.set_xticklabels(
            classes,
            rotation=90,
            fontsize=6
        )

        ax.set_yticklabels(
            classes,
            fontsize=6
        )

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    plt.savefig(
        Path(output_dir) / "confusion_matrix.png",
        dpi=200
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Classical Image Classification"
    )

    parser.add_argument(
        "--data_dir",
        default="dataset"
    )

    parser.add_argument(
        "--output_dir",
        default="results"
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=DEFAULT_MAX_TRAIN_SAMPLES
    )

    parser.add_argument(
        "--force_rebuild",
        action="store_true"
    )

    args = parser.parse_args()

    initialize_gpu()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    classes = discover_classes(data_dir)

    encoder = LabelEncoder()
    encoder.fit(classes)

    with open(
        output_dir / "class_mapping.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                str(i): name
                for i, name in enumerate(encoder.classes_)
            },
            f,
            indent=4
        )

    # --------------------------------------------------------
    # GPU FEATURE EXTRACTION
    # --------------------------------------------------------

    X_train, y_train = extract_split_features(
        data_dir,
        "train",
        classes,
        encoder,
        output_dir,
        args.force_rebuild
    )

    X_valid, y_valid = extract_split_features(
        data_dir,
        "valid",
        classes,
        encoder,
        output_dir,
        args.force_rebuild
    )

    X_test, y_test = extract_split_features(
        data_dir,
        "test",
        classes,
        encoder,
        output_dir,
        args.force_rebuild
    )

    print("\nDATASET SUMMARY")
    print("-" * 50)
    print("Classes:", len(classes))
    print("Train:", len(y_train))
    print("Valid:", len(y_valid))
    print("Test:", len(y_test))
    print("Features:", TOTAL_FEATURES)

    # --------------------------------------------------------
    # TRAINING SAMPLE
    # --------------------------------------------------------

    indices = select_training_indices(
        y_train,
        args.max_train_samples
    )

    y_selected = np.asarray(
        y_train[indices]
    )

    # --------------------------------------------------------
    # EXPERIMENTS
    # --------------------------------------------------------

    experiments = build_experiments()

    results = []

    best_score = -1
    best_model = None
    best_name = None
    best_type = None
    best_slice = None

    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    for name, config in experiments.items():

        print("\n" + "=" * 65)
        print("EXPERIMENT:", name)
        print("FEATURES:", config["features"])
        print("MODEL:", config["model"])
        print("=" * 65)

        feature_slice = config["features"]
        model_type = config["model"]

        model = create_model(model_type)

        X_selected = np.asarray(
            X_train[indices, feature_slice],
            dtype=np.float32
        )

        print("Training samples:", len(X_selected))
        print("Feature dimensions:", X_selected.shape[1])

        if model_type.startswith("gpu_"):
            print("Training device: NVIDIA GPU")
        else:
            print("Training device: CPU")

        model.fit(
            X_selected,
            y_selected
        )

        # Release the large feature sample after fitting.
        del X_selected
        gc.collect()

        validation_metrics, _, validation_report = evaluate(
            model,
            X_valid,
            y_valid,
            model_type,
            feature_slice,
            classes,
            "Validation"
        )

        validation_report.to_csv(
            output_dir / f"{name}_validation_report.csv"
        )

        results.append({
            "Experiment": name,
            "Device": (
                "GPU" if model_type.startswith("gpu_") else "CPU"
            ),
            **validation_metrics
        })

        if validation_metrics["Macro F1"] > best_score:

            best_score = validation_metrics["Macro F1"]
            best_model = model
            best_name = name
            best_type = model_type
            best_slice = feature_slice

            print("\nNew best model:", best_name)

        # Save each model.
        joblib.dump(
            {
                "model": model,
                "model_type": model_type,
                "feature_start": feature_slice.start,
                "feature_stop": feature_slice.stop,
                "experiment": name,
                "classes": classes,
                "image_size": IMAGE_SIZE
            },
            model_dir / f"{name}.joblib"
        )

        del model
        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    comparison = save_comparison(
        results,
        output_dir
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    print("\nBest model:", best_name)
    print("Validation Macro F1:", round(best_score, 4))

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    best_model_path = model_dir / "best_model.joblib"

    joblib.dump(
        {
            "model": best_model,
            "model_type": best_type,
            "feature_start": best_slice.start,
            "feature_stop": best_slice.stop,
            "experiment": best_name,
            "classes": classes,
            "image_size": IMAGE_SIZE
        },
        best_model_path
    )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    test_metrics, test_predictions, test_report = evaluate(
        best_model,
        X_test,
        y_test,
        best_type,
        best_slice,
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
    ) as f:

        json.dump(
            test_metrics,
            f,
            indent=4
        )

    save_confusion_matrix(
        y_test,
        test_predictions,
        classes,
        output_dir
    )

    # --------------------------------------------------------
    # EXPERIMENT SUMMARY
    # --------------------------------------------------------

    summary = {
        "project": "GPU-Accelerated Classical Multi-Class Image Classification",
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "number_of_classes": len(classes),
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "hog_features": HOG_FEATURES,
        "lbp_features": LBP_FEATURES,
        "glcm_features": GLCM_FEATURES,
        "total_features": TOTAL_FEATURES,
        "train_images": len(y_train),
        "valid_images": len(y_valid),
        "test_images": len(y_test),
        "training_samples_per_experiment": len(indices),
        "best_model": best_name,
        "validation_macro_f1": best_score,
        "test_metrics": test_metrics
    }

    with open(
        output_dir / "experiment_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n" + "=" * 65)
    print("CLASSIFICATION COMPLETED")
    print("=" * 65)

    print("Best model:", best_name)
    print("Best model file:", best_model_path)
    print("Results:", output_dir.resolve())


if __name__ == "__main__":
    main()
