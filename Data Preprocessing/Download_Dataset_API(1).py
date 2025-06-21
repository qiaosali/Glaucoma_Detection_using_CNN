import kaggle
import zipfile
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

kaggle.api.authenticate()
kaggle.api.dataset_download_files('sshikamaru/glaucoma-detection', path='.', unzip=False)

zip_file_path = 'F:/Thesis/Project_PyQt6/pyqt6/datasets/Sample_Data/glaucoma-detection.zip'
output_dir = 'F:/Thesis/Project_PyQt6/pyqt6/datasets/Sample_Data/glaucoma-detection.zip/glaucoma_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Check if the zip file exists
if os.path.exists(zip_file_path):
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Kaggle API is working and the dataset has been unzipped!")
else:
    print(f"Error: The file {zip_file_path} does not exist. Please check the download process.")





