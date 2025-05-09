import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings(action='ignore')

# Paths
# yes_images_folder = r"F:/datasets/organized_data/yes/images"
# output_dir = r"F:/datasets/organized_data/aug_yes"
# aug_images_folder = os.path.join(output_dir, "images")
# aug_csv_folder = os.path.join(output_dir, "csv")
# yes_csv_path = r"F:/datasets/organized_data/yes/csv/data.csv"

yes_images_folder = r"F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/yes/images"
output_dir = r"F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/aug_yes"
aug_images_folder = os.path.join(output_dir, "images")
aug_csv_folder = os.path.join(output_dir, "csv")
yes_csv_path = r"F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/yes/csv/data.csv"

# Load the CSV file
yes_data = pd.read_csv(yes_csv_path)

# Create output directories for augmented data
os.makedirs(aug_images_folder, exist_ok=True)
os.makedirs(aug_csv_folder, exist_ok=True)

# Define augmentation transformations
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# List to store new rows for CSV
augmented_rows = []

# Augment Images and Create New Data
for _, row in yes_data.iterrows():
    filename = row["Filename"]
    src_path = os.path.join(yes_images_folder, filename)

    if os.path.exists(src_path):
        # Load the original image
        img = Image.open(src_path)
        img_array = np.expand_dims(np.array(img), axis=0)

        # Save the original image to the new folder
        new_filename = filename
        new_path = os.path.join(aug_images_folder, new_filename)
        img.save(new_path)

        # Add the original image row to augmented_rows
        new_row = row.copy()
        new_row["Filename"] = new_filename
        augmented_rows.append(new_row)

        # Generate 2 augmented images
        i = 1
        for batch in datagen.flow(img_array, batch_size=1):
            augmented_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            augmented_image_path = os.path.join(aug_images_folder, augmented_filename)

            # Save the augmented image
            augmented_img = Image.fromarray(batch[0].astype("uint8"))
            augmented_img.save(augmented_image_path)

            # Add new augmented image row to augmented_rows
            new_augmented_row = row.copy()
            new_augmented_row["Filename"] = augmented_filename
            augmented_rows.append(new_augmented_row)

            i += 1
            if i > 2:  # We need exactly 2 augmented images (total of 3 images: original + 2 augmented)
                break

# Save updated CSV with 3 images (original + 2 augmented) for each entry
augmented_data = pd.DataFrame(augmented_rows)
augmented_data.to_csv(os.path.join(aug_csv_folder, "data.csv"), index=False)

print(f"Augmented images saved to '{aug_images_folder}' and CSV updated at '{aug_csv_folder}/data.csv'.")



