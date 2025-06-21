import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

file_path = "F:\datasets\ORIGA\ORIGA\Images"

# List all files in the directory
all_files = os.listdir(file_path)

jpg_files = [file for file in all_files if file.lower().endswith(".jpg")]

# Count the number of images
print("Number of images:", len(jpg_files))

file_path = "F:\Thesis\Project_PyQt6\pyqt6\datasets\Sample_Data\glaucoma.csv"

data = pd.read_csv(file_path)
print("Number of rows:", data.shape[0])
print("Number of columns:",data.shape[1])
print("Column names:", data.columns.tolist())

for column in data.columns:
    print(f"Unique values in '{column}':")
    print(data[column].unique(), "\n")

print(data.info())
print(data.isna().sum())







