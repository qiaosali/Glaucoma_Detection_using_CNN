import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split

import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings(action='ignore')

# Create directory if it doesn't exist
save_dir = 'F:/Thesis/Project_PyQt6/pyqt6/datasets/working'
os.makedirs(save_dir, exist_ok=True)

# Update save paths for EfficientNetB0
weights_path = os.path.join(save_dir, 'F:/Thesis/Project_PyQt6/pyqt6/Efficient-Net Model/efficient0_model_weights.weights.h5')
model_path = os.path.join(save_dir, 'F:/Thesis/Project_PyQt6/pyqt6/Efficient-Net Model/efficient0_model.h5')

# Debug paths
print(f"Saving weights to: {weights_path}")
print(f"Saving model to: {model_path}")

# Create the model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # Assuming binary classification
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Now save weights and model
model.save_weights(weights_path)
model.save(model_path)
print("EfficientNetB0 Model and weights saved successfully!")

# Rest of your validation code...
validation_path_dir = 'F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/yes'

if not os.path.exists(validation_path_dir):
    raise FileNotFoundError(f"The directory {validation_path_dir} does not exist.")

print("Contents of validation directory:")
for root, dirs, files in os.walk(validation_path_dir):
    print(f"{root}: {dirs} {files}")

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_path_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"Number of validation samples: {validation_generator.samples}")

if validation_generator.samples > 0:
    results = model.evaluate(validation_generator)
    print(f"Validation Loss: {results[0]}, Validation Accuracy: {results[1]}")
else:
    print("No validation samples found. Please check the validation directory.")