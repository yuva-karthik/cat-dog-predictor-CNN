import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_dataset_dir = r"C:\Users\satellite\OneDrive\Desktop\image detection\datasets\PetImages"
base_dir = r"C:\Users\satellite\OneDrive\Desktop\image detection\datasets"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

os.makedirs(train_dir + "/cats", exist_ok=True)
os.makedirs(train_dir + "/dogs", exist_ok=True)
os.makedirs(validation_dir + "/cats", exist_ok=True)
os.makedirs(validation_dir + "/dogs", exist_ok=True)

# Split Data
def split_data(category):
    src_dir = os.path.join(original_dataset_dir, category.capitalize())
    images = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    for img_file in train_images:
        src = os.path.join(src_dir, img_file)
        dst = os.path.join(train_dir, category + "s", img_file)
        shutil.copyfile(src, dst)

    for img_file in val_images:
        src = os.path.join(src_dir, img_file)
        dst = os.path.join(validation_dir, category + "s", img_file)
        shutil.copyfile(src, dst)

split_data('cat')
split_data('dog')

print("Dataset organized successfully!")
