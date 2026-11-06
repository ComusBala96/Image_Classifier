
from pathlib import Path
import cv2
import csv
import yaml
import json
import shutil
from collections import Counter

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SOURCE = Path("./original")
OUTPUT = Path("./dataset_original")

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
]

# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

yaml_path = SOURCE / "data.yaml"

if not yaml_path.exists():
    raise FileNotFoundError(f"Missing YAML: {yaml_path}")

with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, list):
    class_names = {
        i: name for i, name in enumerate(names)
    }
else:
    class_names = {
        int(k): v for k, v in names.items()
    }

print("Detected classes:", class_names)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def find_image(folder, stem):
    for ext in IMAGE_EXTENSIONS:
        path = folder / (stem + ext)
        if path.exists():
            return path
    return None


# --------------------------------------------------
# IMAGE-LEVEL CONVERSION
# --------------------------------------------------

manifest = []
errors = []
summary = {
    split: Counter() for split in SPLITS
}

for split in SPLITS:

    image_folder = SOURCE / "images" / split
    label_folder = SOURCE / "labels" / split

    if not image_folder.exists():
        print(f"Missing: {image_folder}")
        continue

    if not label_folder.exists():
        print(f"Missing: {label_folder}")
        continue

    print(f"\nProcessing {split}...")

    # Process images, not label files, so images
    # without annotations are also detected.
    image_files = [
        p for p in image_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image_path in image_files:

        label_path = label_folder / (
            image_path.relative_to(image_folder).with_suffix(".txt")
        )

        if not label_path.exists():
            errors.append({
                "split": split,
                "image": str(image_path),
                "error": "Missing label file"
            })
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            errors.append({
                "split": split,
                "image": str(image_path),
                "error": "Unreadable image"
            })
            continue

        height, width = image.shape[:2]

        # Aggregate bounding-box area by class.
        class_areas = Counter()
        annotated_classes = set()

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:

            values = line.strip().split()

            if len(values) != 5:
                errors.append({
                    "split": split,
                    "image": str(image_path),
                    "error": "Invalid annotation line"
                })
                continue

            try:
                class_id = int(float(values[0]))
                xc, yc, bw, bh = map(float, values[1:])
            except ValueError:
                continue

            if class_id not in class_names:
                continue

            if not (
                0 <= xc <= 1 and 0 <= yc <= 1
                and 0 < bw <= 1 and 0 < bh <= 1
            ):
                continue

            # Normalized bounding-box area
            area = bw * bh

            class_areas[class_id] += area
            annotated_classes.add(class_id)

        if not class_areas:
            errors.append({
                "split": split,
                "image": str(image_path),
                "error": "No valid annotations"
            })
            continue

        # Dominant class = largest total annotated area
        selected_class = max(
            class_areas,
            key=class_areas.get
        )

        class_name = class_names[selected_class]

        destination_folder = OUTPUT / split / class_name
        destination_folder.mkdir(
            parents=True, exist_ok=True
        )

        destination_path = destination_folder / image_path.name

        # Avoid silently overwriting files
        if destination_path.exists():
            errors.append({
                "split": split,
                "image": str(image_path),
                "error": "Destination filename collision"
            })
            continue

        shutil.copy2(image_path, destination_path)

        summary[split][class_name] += 1

        manifest.append({
            "split": split,
            "original_image": str(image_path),
            "label_file": str(label_path),
            "class_id": selected_class,
            "class_name": class_name,
            "annotated_class_count": len(annotated_classes),
            "multiple_classes": len(annotated_classes) > 1,
            "copied_image": str(destination_path)
        })

# --------------------------------------------------
# SAVE REPORTS
# --------------------------------------------------

OUTPUT.mkdir(parents=True, exist_ok=True)

if manifest:
    with open(
        OUTPUT / "dataset_manifest.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=manifest[0].keys()
        )
        writer.writeheader()
        writer.writerows(manifest)

summary_json = {
    split: dict(counts)
    for split, counts in summary.items()
}

with open(
    OUTPUT / "conversion_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary_json, f, indent=4)

with open(
    OUTPUT / "conversion_errors.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(errors, f, indent=4)

# --------------------------------------------------
# DISPLAY SUMMARY
# --------------------------------------------------

print("\n========== IMAGE CONVERSION SUMMARY ==========")

for split in SPLITS:
    print(f"\n{split.upper()}")

    for class_name in class_names.values():
        print(
            f"{class_name}: "
            f"{summary[split][class_name]}"
        )

    print("Total:", sum(summary[split].values()))

multi_class_count = sum(
    item["multiple_classes"] for item in manifest
)

print("\nTotal copied images:", len(manifest))
print("Images with multiple annotated classes:", multi_class_count)
print("Errors:", len(errors))
print("Output:", OUTPUT.resolve())
print("\nConversion completed.")
