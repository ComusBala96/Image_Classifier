import os

# Main parent folder
parent_folder = r"D:\Projects\Image_Classifier\original"

# Supported file extensions
extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Loop through all subfolders
for folder_name in os.listdir(parent_folder):

    folder_path = os.path.join(parent_folder, folder_name)

    if not os.path.isdir(folder_path):
        continue

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    ])

    for index, old_name in enumerate(files, start=1):

        extension = os.path.splitext(old_name)[1]

        new_name = f"{folder_name}_{index:05d}{extension}"

        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

    print(f"{folder_name}: {len(files)} files renamed")

print("\nAll folders processed successfully!")