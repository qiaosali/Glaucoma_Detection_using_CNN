import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import EfficientNetB0  # Added missing import
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import warnings
# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings(action='ignore')

# Define U-Net model architecture first
def unet_model(input_shape=(224, 224, 3)):
    inputs = Input(input_shape)

    # Downsample path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)

    # Upsample path
    u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)

    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and compile the model
model = unet_model()

# Create directory if it doesn't exist
save_dir = 'F:/Thesis/Project_PyQt6/pyqt6/datasets/working'
os.makedirs(save_dir, exist_ok=True)

# Ensure file paths are valid
weights_path = 'F:/Thesis/Project_PyQt6/pyqt6/U-Net Model/unet_model_weights.weights.h5'
model_path = 'F:/Thesis/Project_PyQt6/pyqt6/U-Net Model/unet_model.h5'

# Debug paths
print(f"Saving weights to: {weights_path}")
print(f"Saving model to: {model_path}")

# Create the EfficientNet model
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

# Save weights and model
model.save_weights(weights_path)
model.save(model_path)

print("Model and weights saved successfully!")

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