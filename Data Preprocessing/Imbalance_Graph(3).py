import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')

# Imbalance Graph
data = pd.read_csv('F:/Thesis/Project_PyQt6/pyqt6/datasets/Sample_Data/glaucoma.csv')

# Count the occurrences of each class in the 'Glaucoma' column
glaucoma_counts = data['Glaucoma'].value_counts()

# Print the counts to check for imbalance
print("Glaucoma distribution:")
print(glaucoma_counts)

# Plotting the bar chart to visualize the imbalance with specified colors
plt.figure(figsize=(6, 4))
sns.barplot(x=glaucoma_counts.index, y=glaucoma_counts.values, palette=['blue', 'green'])
plt.title('Distribution of Glaucoma Classes')
plt.xlabel('Glaucoma (0: No, 1: Yes)')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
plt.show()


# Paths
base_dir = "F:/datasets/ORIGA/ORIGA"
csv_file_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/Sample_Data/glaucoma.csv"
image_dir = os.path.join(base_dir, "Images")
output_dir = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data"

# Load the CSV file
data = pd.read_csv(csv_file_path)

# Create output directories for 'yes' and 'no'
yes_folder = os.path.join(output_dir, "yes")
no_folder = os.path.join(output_dir, "no")

# Subfolders for images and CSVs
yes_images_folder = os.path.join(yes_folder, "images")
yes_csv_folder = os.path.join(yes_folder, "csv")
no_images_folder = os.path.join(no_folder, "images")
no_csv_folder = os.path.join(no_folder, "csv")

# Create all necessary folders
os.makedirs(yes_images_folder, exist_ok=True)
os.makedirs(yes_csv_folder, exist_ok=True)
os.makedirs(no_images_folder, exist_ok=True)
os.makedirs(no_csv_folder, exist_ok=True)

# Separate data based on 'Glaucoma' values
yes_data = data[data["Glaucoma"] == 1]
no_data = data[data["Glaucoma"] == 0]

# Function to organize images and save CSVs
def organize_data(subset_data, images_folder, csv_folder):
    for _, row in subset_data.iterrows():
        filename = row["Filename"]
        src_path = os.path.join(image_dir, filename)
        dest_path = os.path.join(images_folder, filename)

        # Copy image
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)

    # Save corresponding CSV in the 'csv' subfolder
    csv_path = os.path.join(csv_folder, "data.csv")
    subset_data.to_csv(csv_path, index=False)

# Organize data for 'yes' and 'no' classes
organize_data(yes_data, yes_images_folder, yes_csv_folder)
organize_data(no_data, no_images_folder, no_csv_folder)

print("Data organized into 'yes' and 'no' folders with separate subfolders for images and CSVs!/n")

no_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/no/images"
yes_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/yes/images"

# List all files in the directory
nofiles = os.listdir(no_path)
yesfiles = os.listdir(yes_path)

no_files = [file for file in nofiles if file.lower().endswith(".jpg")]
yes_files = [file for file in yesfiles if file.lower().endswith(".jpg")]
# Count the number of images
print("Number of images in no directory:", len(no_files))
print("Number of images in yes directory:", len(yes_files))
