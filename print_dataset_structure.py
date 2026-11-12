
from pathlib import Path

DATASET_DIR = Path("dataset")

def print_structure(folder, prefix=""):
    folders = sorted(
        [p for p in folder.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )

    for index, item in enumerate(folders):
        is_last = index == len(folders) - 1

        connector = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")

        print(f"{prefix}{connector}{item.name}/")
        print_structure(item, next_prefix)


if __name__ == "__main__":
    if DATASET_DIR.is_dir():
        print(f"{DATASET_DIR.name}/")
        print_structure(DATASET_DIR)
    else:
        print("Dataset folder not found:", DATASET_DIR.resolve())
