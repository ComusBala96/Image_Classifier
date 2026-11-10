
import os
import random
import shutil
from pathlib import Path

# Dataset configuration
SOURCE_DIR = Path("./original")
OUTPUT_DIR = Path("./dataset")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"
}

random.seed(RANDOM_SEED)


def split_dataset():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    if abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Split ratios must add up to 1.0")

    # Create output directories
    for split in ["train", "valid", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    total_images = 0

    print("\n========== DATASET SPLITTING ==========\n")

    # Process each class folder
    for class_folder in sorted(SOURCE_DIR.iterdir()):

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        images = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        total = len(images)

        # Calculate split counts
        train_count = int(total * TRAIN_RATIO)
        valid_count = int(total * VALID_RATIO)
        test_count = total - train_count - valid_count

        # Split images
        train_images = images[:train_count]
        valid_images = images[train_count:train_count + valid_count]
        test_images = images[train_count + valid_count:]

        splits = {
            "train": train_images,
            "valid": valid_images,
            "test": test_images
        }

        # Copy images into respective folders
        for split_name, split_images in splits.items():

            destination = OUTPUT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for image_file in split_images:
                shutil.copy2(
                    image_file,
                    destination / image_file.name
                )

        total_images += total

        print(f"Class: {class_name}")
        print(f"  Total: {total}")
        print(f"  Train: {len(train_images)}")
        print(f"  Valid: {len(valid_images)}")
        print(f"  Test : {len(test_images)}")
        print("-----------------------------------")

    print("\n========== FINAL SUMMARY ==========")
    print(f"Total Images: {total_images}")

    for split in ["train", "valid", "test"]:

        split_path = OUTPUT_DIR / split

        count = sum(
            1 for f in split_path.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

        percentage = (count / total_images * 100) if total_images else 0

        print(f"{split.upper():8}: {count:6} ({percentage:.2f}%)")

    print("\nDataset splitting completed successfully!")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    split_dataset()
