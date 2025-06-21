import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings(action='ignore')

file_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/aug_yes/images"

# List all files in the directory
all_files = os.listdir(file_path)

aug_files = [file for file in all_files if file.lower().endswith(".jpg")]
# Count the number of images
print("Number of total augmented images:", len(aug_files))

file_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/aug_yes/csv/data.csv"

df = pd.read_csv(file_path)
print("Number of rows:", df.shape[0])
print("Number of columns:",df.shape[1])
print("Column names:", df.columns.tolist())

# Paths to the image folders
aug_yes_images_folder = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/aug_yes/images"
no_images_folder = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/no/images"

# Count the number of images in each folder
num_aug_yes_images = len([img for img in os.listdir(aug_yes_images_folder) if img.endswith(".jpg")])
num_no_images = len([img for img in os.listdir(no_images_folder) if img.endswith(".jpg")])

# Data for the bar chart
categories = ['Augmented Yes', 'No']
counts = [num_aug_yes_images, num_no_images]

# Plot the bar chart
plt.figure(figsize=(8, 5))
plt.bar(categories, counts, color=['blue', 'green'])
plt.title('Comparison of Image Counts Between Augmented Yes and No Folders')
plt.xlabel('Category')
plt.ylabel('Number of Images')
plt.grid(False)
plt.show()

print("Data is balanced")





