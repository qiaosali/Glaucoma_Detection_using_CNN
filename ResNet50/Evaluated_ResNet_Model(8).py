import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler  # Import LearningRateScheduler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings(action='ignore')

# Create directory if it doesn't exist
save_dir = 'F:/datasets/working'

os.makedirs(save_dir, exist_ok=True)

# Ensure file paths are valid
weights_path = 'F:/Thesis/Project_PyQt6/pyqt6/ResNet50 Model/resnet50_model_weights.weights.h5'
model_path = 'F:/Thesis/Project_PyQt6/pyqt6/ResNet50 Model/resnet50_model.h5'


# Debug paths
print(f"Saving weights to: {weights_path}")
print(f"Saving model to: {model_path}")

# Load the base ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# predictions = Dense(10, activation='softmax')(x)  # Assuming 10 classes
predictions = Dense(2, activation='softmax')(x)  # Assuming 2 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save weights and model
model.save_weights(weights_path)
model.save(model_path)

print("Model and weights saved successfully!")

# Define the path to your validation data
validation_path_dir= 'F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/yes' #168 images

# Check if the directory exists
if not os.path.exists(validation_path_dir):
    raise FileNotFoundError(f"The directory {validation_path_dir} does not exist.")

# Print the contents of the directory
print("Contents of validation directory:")
for root, dirs, files in os.walk(validation_path_dir):
    print(f"{root}: {dirs} {files}")

# Create an ImageDataGenerator for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load the validation data Or #Set up the validation generator
validation_generator = validation_datagen.flow_from_directory(
    # validation_data_dir,
    validation_path_dir,
    # r'F:/datasets/working/validation/resnet50_model.h5',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Print the number of samples in the validation generator
print(f"Number of validation samples: {validation_generator.samples}")

# Evaluate the model
if validation_generator.samples > 0:
    results = model.evaluate(validation_generator)
    print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")
else:
    print("No validation samples found. Please check the validation directory.")

