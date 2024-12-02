import os
import shutil

# Define the source and destination directories
source_dir = os.getcwd()  # Current working directory
destination_dir = os.path.join(source_dir, "copied_scripts")

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Walk through the directory structure
for root, dirs, files in os.walk(source_dir):
    # Exclude the destination directory from the copy
    if destination_dir.startswith(root):
        continue

    # Create corresponding directories in the destination
    relative_path = os.path.relpath(root, source_dir)
    target_path = os.path.join(destination_dir, relative_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Copy files with .py extension as .txt
    for file in files:
        if file.endswith(".py"):
            source_file_path = os.path.join(root, file)
            destination_file_path = os.path.join(target_path, file[:-3] + ".txt")
            shutil.copy2(source_file_path, destination_file_path)

print("All .py files have been copied and renamed with a .txt extension.")
