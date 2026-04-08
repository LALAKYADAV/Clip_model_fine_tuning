import kagglehub
import shutil
import os

# Step 1: Download dataset (goes to kaggle cache)
src_path = kagglehub.dataset_download("yakhyokhuja/ms1m-arcface-dataset")

print("Downloaded to:", src_path)

# Step 2: Define destination
dst_path = r"D:\clip\clip_clip\data\ms1m-arcface"

# Step 3: Move dataset
if not os.path.exists(dst_path):
    shutil.move(src_path, dst_path)
else:
    print("Destination already exists!")

print("Dataset moved to:", dst_path)