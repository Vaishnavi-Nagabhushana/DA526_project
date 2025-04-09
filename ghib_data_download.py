import kagglehub
import shutil
import os

# Download the dataset explicitly
path = kagglehub.dataset_download("shubham1921/real-to-ghibli-image-dataset-5k-paired-images")
print("Dataset downloaded at:", path)

# explicitly define output path
destination = r'D:/ghibli_data'

# Ensure the destination directory exists
os.makedirs(destination, exist_ok=True)

# Explicitly copy the contents of the downloaded dataset to the destination folder
for item in os.listdir(path):
    source_item = os.path.join(path, item)
    dest_item = os.path.join(destination, item)

    # Directory copy explicitly
    if os.path.isdir(source_item):
        shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
        print(f"Copied directory explicitly clearly: {source_item} → {dest_item}")

    # File copy explicitly clearly
    else:
        shutil.copy2(source_item, dest_item)
        print(f"Copied file explicitly clearly: {source_item} → {dest_item}")

print( "Ghibli dataset clearly and explicitly successfully saved at:", destination)