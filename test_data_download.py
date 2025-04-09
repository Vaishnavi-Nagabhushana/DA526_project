import kagglehub
import os
import shutil

# download explicitly via kagglehub
downloaded_path = kagglehub.dataset_download("labledata/ghibli-dataset")
print("Dataset explicitly downloaded at:", downloaded_path)

# Define explicitly source path (change it to the actual downloaded path)
source_dir = os.path.join(downloaded_path, "Ghibli")

# Define explicitly your final destination explicitly clearly
destination_dir = "D:/ghibli_test"

# explicitly clear creation of direct structure explicitly
real_dest = os.path.join(destination_dir, "real")
ghibli_dest = os.path.join(destination_dir, "ghibli")

# Create destination folders explicitly clearly
os.makedirs(real_dest, exist_ok=True)
os.makedirs(ghibli_dest, exist_ok=True)

# Loop explicitly clearly through each split explicitly
for split in ["testing", "training", "validation"]:
    split_path = os.path.join(source_dir, split)
    
    for folder in sorted(os.listdir(split_path)):
        curr_folder_path = os.path.join(split_path, folder)

        # clearly define explicitly source images explicitly clearly explicitly
        real_img = os.path.join(curr_folder_path, "o.png")  # original explicitly explicitly real explicitly explicitly image explicitly
        style_img = os.path.join(curr_folder_path, "g.png") # cartoon explicitly explicitly ghibli explicitly style explicitly explicitly

        real_dest_path = os.path.join(real_dest, f"{split}_{folder}.png")
        style_dest_path = os.path.join(ghibli_dest, f"{split}_{folder}.png")

        # explicitly copying explicitly clearly images explicitly explicitly clearly
        shutil.copy(real_img, real_dest_path)
        shutil.copy(style_img, style_dest_path)

print("\n data at:", destination_dir)