from datasets import load_dataset
import os
from PIL import Image

# ✅ Load dataset
ds = load_dataset("dqymaggie/cartoonizer-dataset")

# ✅ Perform Hugging Face's split (75% train, 25% test)
split_ds = ds["train"].train_test_split(test_size=0.25, seed=42)
train_data = split_ds["train"]
test_data = split_ds["test"]

# ✅ Define output directories
root_dir = "/home/vaishnavi/main/DA526_Project/datasets"
train_content_dir = os.path.join(root_dir, "train", "realworld")
train_style_dir = os.path.join(root_dir, "train", "cartoon")
test_content_dir = os.path.join(root_dir, "test", "realworld")
test_style_dir = os.path.join(root_dir, "test", "cartoon")

# ✅ Create directories
for directory in [train_content_dir, train_style_dir, test_content_dir, test_style_dir]:
    os.makedirs(directory, exist_ok=True)

# ✅ Save train images
print("Saving training images:")
for i, entry in enumerate(train_data):
    entry["original_image"].save(os.path.join(train_content_dir, f"real_{i+1:05d}.png"))
    entry["cartoonized_image"].save(os.path.join(train_style_dir, f"cartoon_{i+1:05d}.png"))

    if (i + 1) % 500 == 0:
        print(f"✅ Saved {i + 1} train pairs.")

# ✅ Save test images
print("\nSaving testing images:")
for i, entry in enumerate(test_data):
    entry["original_image"].save(os.path.join(test_content_dir, f"real_{i+1:05d}.png"))
    entry["cartoonized_image"].save(os.path.join(test_style_dir, f"cartoon_{i+1:05d}.png"))

    if (i + 1) % 500 == 0:
        print(f"✅ Saved {i + 1} test pairs.")

print("\n🎉 Dataset split and saved successfully!")
