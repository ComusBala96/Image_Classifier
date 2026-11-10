from pathlib import Path
import cv2
import csv
import yaml
import json

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SOURCE = Path(r"./original")
OUTPUT = Path(r"./dataset")

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Add 5% padding around each bounding box
PADDING = 0.05

# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

yaml_path = SOURCE / "data.yaml"

if not yaml_path.exists():
    raise FileNotFoundError(
        f"Cannot find {yaml_path}. Check your dataset YAML path."
    )

with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, list):
    class_names = {i: name for i, name in enumerate(names)}
else:
    class_names = {int(k): v for k, v in names.items()}

print("Detected classes:")
for class_id, name in class_names.items():
    print(f"{class_id}: {name}")

# --------------------------------------------------
# HELPER: FIND MATCHING IMAGE
# --------------------------------------------------

def find_image(image_folder, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = image_folder / (stem + extension)
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------
# CONVERSION
# --------------------------------------------------

manifest = []
summary = {
    split: {name: 0 for name in class_names.values()}
    for split in SPLITS
}

errors = []

for split in SPLITS:

    image_folder = SOURCE / "images" / split
    label_folder = SOURCE / "labels" / split

    if not image_folder.exists():
        print(f"WARNING: Missing image folder: {image_folder}")
        continue

    if not label_folder.exists():
        print(f"WARNING: Missing label folder: {label_folder}")
        continue

    label_files = list(label_folder.glob("*.txt"))

    print(f"\nProcessing {split}: {len(label_files)} label files")

    for label_file in label_files:

        image_path = find_image(image_folder, label_file.stem)

        if image_path is None:
            errors.append({
                "split": split,
                "file": label_file.name,
                "error": "Matching image not found"
            })
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            errors.append({
                "split": split,
                "file": image_path.name,
                "error": "Cannot read image"
            })
            continue

        height, width = image.shape[:2]

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for object_index, line in enumerate(lines):

            values = line.strip().split()

            if len(values) != 5:
                errors.append({
                    "split": split,
                    "file": label_file.name,
                    "error": f"Invalid YOLO annotation: {line.strip()}"
                })
                continue

            try:
                class_id = int(float(values[0]))
                xc, yc, bw, bh = map(float, values[1:])
            except ValueError:
                continue

            if class_id not in class_names:
                errors.append({
                    "split": split,
                    "file": label_file.name,
                    "error": f"Unknown class ID: {class_id}"
                })
                continue

            if not (
                0 <= xc <= 1 and 0 <= yc <= 1
                and 0 < bw <= 1 and 0 < bh <= 1
            ):
                errors.append({
                    "split": split,
                    "file": label_file.name,
                    "error": "Invalid bounding-box coordinates"
                })
                continue

            # Convert normalized coordinates to pixels
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            # Add padding
            pad_x = int((x2 - x1) * PADDING)
            pad_y = int((y2 - y1) * PADDING)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            class_name = class_names[class_id]

            # Create class directory
            destination_folder = (
                OUTPUT / split / class_name
            )
            destination_folder.mkdir(
                parents=True, exist_ok=True
            )

            crop_filename = (
                f"{image_path.stem}_obj{object_index:03d}.jpg"
            )

            crop_path = destination_folder / crop_filename

            success = cv2.imwrite(str(crop_path), crop)

            if not success:
                continue

            summary[split][class_name] += 1

            manifest.append({
                "split": split,
                "original_image": str(image_path),
                "label_file": str(label_file),
                "class_id": class_id,
                "class_name": class_name,
                "object_index": object_index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_path": str(crop_path)
            })

# --------------------------------------------------
# SAVE REPORTS
# --------------------------------------------------

OUTPUT.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT / "dataset_manifest.csv"

if manifest:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=manifest[0].keys()
        )
        writer.writeheader()
        writer.writerows(manifest)

with open(OUTPUT / "conversion_summary.json", "w",
          encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

with open(OUTPUT / "conversion_errors.json", "w",
          encoding="utf-8") as f:
    json.dump(errors, f, indent=4)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========== CONVERSION SUMMARY ==========")

for split, classes in summary.items():
    print(f"\n{split.upper()}")

    for class_name, count in classes.items():
        print(f"  {class_name}: {count}")

    print(f"  Total: {sum(classes.values())}")

print("\nTotal extracted objects:", len(manifest))
print("Total errors:", len(errors))
print("Output:", OUTPUT)
print("Manifest:", csv_path)
print("\nConversion completed.")
