# import os
# import shutil
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# from tensorflow.keras.utils import to_categorical
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate
# from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dense, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.utils import to_categorical
#
# # Disable oneDNN custom operations
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# warnings.filterwarnings(action='ignore')
#
# # Paths for aug_yes and no
# aug_yes_csv_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/aug_yes/csv/data.csv"
# aug_yes_images_folder = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/aug_yes/images"
#
# no_csv_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/no/csv/data.csv"
# no_images_folder = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/no/images"
#
# # Load CSVs
# data_yes = pd.read_csv(aug_yes_csv_path)
# data_yes['label'] = 1  # Assign label 1 for "yes"
#
# data_no = pd.read_csv(no_csv_path)
# data_no['label'] = 0  # Assign label 0 for "no"
#
# # Combine both classes and shuffle
# data = pd.concat([data_yes, data_no], ignore_index=True)
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Split data into train, validation, and test sets (70:15:15) while maintaining label balance
# train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=42)
# valid_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)
#
# # Function to preprocess images and create data lists
# def preprocess_and_load_data(data_subset, folder_paths):
#     images = []
#     labels = []
#
#     for _, row in data_subset.iterrows():
#         filename = row["Filename"]
#         label = row["label"]
#         # Determine the folder based on the label
#         image_folder = folder_paths['yes'] if label == 1 else folder_paths['no']
#         image_path = os.path.join(image_folder, filename)
#
#         if os.path.exists(image_path):
#             # Load the image and resize to (224, 224) for U-Net
#             img = load_img(image_path, target_size=(224, 224))
#             img_array = img_to_array(img)
#             img_preprocessed = preprocess_input(img_array)  # Using the same preprocessing
#
#             images.append(img_preprocessed)
#             labels.append(label)
#
#     return np.array(images), np.array(labels)
#
# # Define image folder paths for both labels
# folder_paths = {
#     "yes": aug_yes_images_folder,
#     "no": no_images_folder
# }
#
# # Preprocess and load data for train, validation, and test sets
# train_images, train_labels = preprocess_and_load_data(train_data, folder_paths)
# valid_images, valid_labels = preprocess_and_load_data(valid_data, folder_paths)
# test_images, test_labels = preprocess_and_load_data(test_data, folder_paths)
#
# # One-hot encode the labels
# num_classes = 2  # Glaucoma has two classes: 0 and 1
# train_labels = to_categorical(train_labels, num_classes=num_classes)
# valid_labels = to_categorical(valid_labels, num_classes=num_classes)
# test_labels = to_categorical(test_labels, num_classes=num_classes)
#
# # Output the shapes of the data
# print(f"Train images shape: {train_images.shape}")
# print(f"Train labels shape: {train_labels.shape}")
# print(f"Validation images shape: {valid_images.shape}")
# print(f"Validation labels shape: {valid_labels.shape}")
# print(f"Test images shape: {test_images.shape}")
# print(f"Test labels shape: {test_labels.shape}")
#
# # Print class distribution in each split
# print("Class distribution in Train:", np.sum(train_labels, axis=0))
# print("Class distribution in Validation:", np.sum(valid_labels, axis=0))
# print("Class distribution in Test:", np.sum(test_labels, axis=0))
#
# # Define data augmentation for the training set
# data_gen = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# # Define U-Net model
# def create_unet_model(input_shape, num_classes):
#     # Input layer
#     inputs = Input(input_shape)
#
#     # Encoder (Contracting Path)
#     # Level 1
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#     # Level 2
#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     # Level 3
#     conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#     # Level 4
#     conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
#     conv4 = BatchNormalization()(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     # Bottleneck
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     # Decoder (Expanding Path)
#     # Level 4
#     up6 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same')(drop5)
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
#     conv6 = BatchNormalization()(conv6)
#
#     # Level 3
#     up7 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(conv6)
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
#     conv7 = BatchNormalization()(conv7)
#
#     # Level 2
#     up8 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(conv7)
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
#     conv8 = BatchNormalization()(conv8)
#
#     # Level 1
#     up9 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(conv8)
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
#     conv9 = BatchNormalization()(conv9)
#
#     # Classification head
#     gap = GlobalAveragePooling2D()(conv9)
#     dense = Dense(64, activation='relu')(gap)
#     outputs = Dense(num_classes, activation='softmax')(dense)
#
#     # Create model
#     model = Model(inputs=inputs, outputs=outputs)
#
#     # Compile model
#     model.compile(optimizer=Adam(learning_rate=1e-4),
#                   loss=tf.losses.CategoricalCrossentropy(),
#                   metrics=['accuracy'])
#
#     return model
#
# def lr_schedule(epoch, lr):
#     # Example: Decrease learning rate by 10% every 5 epochs
#     if epoch % 5 == 0 and epoch > 0:
#         lr *= 0.9
#     return float(lr)  # Ensure the return value is a float
#
# # Create LearningRateScheduler callback
# learning_rate_scheduler = LearningRateScheduler(lr_schedule)
#
# # Callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
#
# # Model configuration
# input_shape = (224, 224, 3)
# num_classes = 2  # Yes and No classes
# model = create_unet_model(input_shape, num_classes)
#
# # Display model summary
# model.summary()
#
# # Training the model
# history = model.fit(
#     data_gen.flow(train_images, train_labels, batch_size=16),  # Smaller batch size for U-Net
#     epochs=8,
#     validation_data=(valid_images, valid_labels),
#     callbacks=[early_stopping, learning_rate_scheduler]
# )
#
# history_df = pd.DataFrame(history.history)
#
# # 2. Set Seaborn style (optional)
# sns.set(style="whitegrid")
#
# # 3. Create a canvas
# plt.figure(figsize=(14, 5))
#
# # 4. Draw the Loss curve
# plt.subplot(1, 2, 1)
# sns.lineplot(data=history_df[['loss', 'val_loss']], palette=['royalblue', 'tomato'], linewidth=2)
# plt.title('Training vs Validation Loss', fontsize=14)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.legend(['Training Loss', 'Validation Loss'], fontsize=10)
#
# # 5. Draw the Accuracy curve
# plt.subplot(1, 2, 2)
# sns.lineplot(data=history_df[['accuracy', 'val_accuracy']], palette=['royalblue', 'tomato'], linewidth=2)
# plt.title('Training vs Validation Accuracy', fontsize=14)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Accuracy', fontsize=12)
# plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=10)
#
# # 6. Adjust layout and display
# plt.tight_layout()
# plt.show()

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings(action='ignore')

# Paths for aug_yes and no
aug_yes_csv_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/aug_yes/csv/data.csv"
aug_yes_images_folder = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/aug_yes/images"

no_csv_path = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/no/csv/data.csv"
no_images_folder = "F:/Thesis/Project_PyQt6/pyqt6/datasets/organized_data/no/images"

# Load CSVs
data_yes = pd.read_csv(aug_yes_csv_path)
data_yes['label'] = 1  # Assign label 1 for "yes"

data_no = pd.read_csv(no_csv_path)
data_no['label'] = 0  # Assign label 0 for "no"

# Combine both classes and shuffle
data = pd.concat([data_yes, data_no], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data into train, validation, and test sets (70:15:15) while maintaining label balance
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Modified function with error handling
def preprocess_and_load_data(data_subset, folder_paths):
    images = []
    labels = []

    for _, row in data_subset.iterrows():
        filename = row["Filename"]
        label = row["label"]
        # Determine the folder based on the label
        image_folder = folder_paths['yes'] if label == 1 else folder_paths['no']
        image_path = os.path.join(image_folder, filename)

        if os.path.exists(image_path):
            try:  # Error handling added here
                # Load the image and resize to (224, 224) for U-Net
                img = load_img(image_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_preprocessed = preprocess_input(img_array)  # Using the same preprocessing

                images.append(img_preprocessed)
                labels.append(label)
            except Exception as e:
                print(f"Warning: Skipped corrupted/invalid image {image_path} - {str(e)}")

    return np.array(images), np.array(labels)

# Define image folder paths for both labels
folder_paths = {
    "yes": aug_yes_images_folder,
    "no": no_images_folder
}

# Preprocess and load data for train, validation, and test sets
train_images, train_labels = preprocess_and_load_data(train_data, folder_paths)
valid_images, valid_labels = preprocess_and_load_data(valid_data, folder_paths)
test_images, test_labels = preprocess_and_load_data(test_data, folder_paths)

# One-hot encode the labels
num_classes = 2  # Glaucoma has two classes: 0 and 1
train_labels = to_categorical(train_labels, num_classes=num_classes)
valid_labels = to_categorical(valid_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Output the shapes of the data
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Validation images shape: {valid_images.shape}")
print(f"Validation labels shape: {valid_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Print class distribution in each split
print("Class distribution in Train:", np.sum(train_labels, axis=0))
print("Class distribution in Validation:", np.sum(valid_labels, axis=0))
print("Class distribution in Test:", np.sum(test_labels, axis=0))

# Define data augmentation for the training set
data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define U-Net model
def create_unet_model(input_shape, num_classes):
    # Input layer
    inputs = Input(input_shape)

    # Encoder (Contracting Path)
    # Level 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Level 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Level 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Level 4
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder (Expanding Path)
    # Level 4
    up6 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    # Level 3
    up7 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    # Level 2
    up8 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    # Level 1
    up9 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Classification head
    gap = GlobalAveragePooling2D()(conv9)
    dense = Dense(64, activation='relu')(gap)
    outputs = Dense(num_classes, activation='softmax')(dense)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=tf.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def lr_schedule(epoch, lr):
    # Example: Decrease learning rate by 10% every 5 epochs
    if epoch % 5 == 0 and epoch > 0:
        lr *= 0.9
    return float(lr)  # Ensure the return value is a float

# Create LearningRateScheduler callback
learning_rate_scheduler = LearningRateScheduler(lr_schedule)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Model configuration
input_shape = (224, 224, 3)
num_classes = 2  # Yes and No classes
model = create_unet_model(input_shape, num_classes)

# Display model summary
model.summary()

# Training the model
history = model.fit(
    data_gen.flow(train_images, train_labels, batch_size=16),  # Smaller batch size for U-Net
    epochs=8,
    validation_data=(valid_images, valid_labels),
    callbacks=[early_stopping, learning_rate_scheduler]
)

history_df = pd.DataFrame(history.history)

# 2. Set Seaborn style (optional)
sns.set(style="whitegrid")

# 3. Create a canvas
plt.figure(figsize=(14, 5))

# 4. Draw the Loss curve
plt.subplot(1, 2, 1)
sns.lineplot(data=history_df[['loss', 'val_loss']], palette=['royalblue', 'tomato'], linewidth=2)
plt.title('Training vs Validation Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=10)

# 5. Draw the Accuracy curve
plt.subplot(1, 2, 2)
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']], palette=['royalblue', 'tomato'], linewidth=2)
plt.title('Training vs Validation Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=10)

# 6. Adjust layout and display
plt.tight_layout()
plt.show()

