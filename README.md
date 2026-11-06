# Weapon Image Classification — Classical Machine Learning

**Author:** Comus Bala  
**Programme:** PhD, Computer Science and Engineering  
**University:** Adamas University  
**Scope:** Classical machine learning only; no deep learning.

## 1. Dataset folder structure

Create a folder named `dataset` and place one subfolder per class:

```text
dataset/
├── Knife/
├── Pistol/
├── Rifle/
├── AK-47/
└── Shotgun/
```

Put the corresponding labelled images inside each class folder. The script supports JPG, JPEG, PNG, BMP, TIFF and WEBP.

## 2. Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Run

```bash
python weapon_classification.py --data_dir dataset --output_dir results
```

Optional arguments:

```bash
python weapon_classification.py --data_dir dataset --output_dir results --test_size 0.15 --cv 3
```

## 4. Experiments

1. HOG + SVM
2. LBP + SVM
3. GLCM + SVM
4. HOG + LBP + SVM
5. HOG + LBP + Random Forest
6. HOG + LBP + KNN
7. HOG + LBP + Decision Tree
8. HOG + LBP + Gaussian Naive Bayes

The script uses a stratified held-out test set. Hyperparameter search and cross-validation happen only on the training partition. Feature scaling is inside the model pipeline to avoid fitting the scaler on test data.

## 5. Outputs

The `results` folder contains:
- `experiment_results.csv` and `.xlsx`
- `classifier_comparison.png`
- `training_time_comparison.png`
- `feature_dimension_comparison.png`
- `confusion_matrices/`
- `models/` (best fitted estimator for each experiment)

## 6. Research integrity and reporting

- The script calculates actual results from the supplied dataset. Do not insert assumed or illustrative accuracy values into the assignment.
- Report the number of images per class and the dataset source/licence.
- If images come from the same video, burst, scene or original source, group them before splitting to prevent near-duplicate leakage.
- Inspect class balance and misclassified examples.
- Report macro-F1 as well as accuracy, particularly if class counts differ.
- Results depend on dataset composition, image quality, preprocessing and random split.
- The eight experiments are a starting comparison, not proof that one classifier is universally best.
- HOG's 8,100-dimensional vector applies to the specified 128×128 configuration (9 orientations, 8×8 cells, 2×2 blocks). LBP uses 10 bins for the specified uniform P=8 configuration.
