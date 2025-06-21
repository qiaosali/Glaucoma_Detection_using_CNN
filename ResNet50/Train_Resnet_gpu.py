# import os
# import shutil
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler  # Import LearningRateScheduler
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from sklearn.model_selection import train_test_split
#
# import warnings
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
#
# ## 修改 数据泄露 问题
# train_data_yes, temp_data_yes = train_test_split(data_yes, test_size=0.3, shuffle=False)
# valid_data_yes, test_data_yes = train_test_split(temp_data_yes, test_size=0.5, shuffle=False)
#
# train_data_no, temp_data_no = train_test_split(data_no, test_size=0.3, shuffle=False)
# valid_data_no, test_data_no = train_test_split(temp_data_no, test_size=0.5, shuffle=False)
#
# train_data = pd.concat([train_data_yes, train_data_no], ignore_index=True)
#
# valid_data = pd.concat([valid_data_yes, valid_data_no], ignore_index=True)
#
# test_data = pd.concat([test_data_yes, test_data_no], ignore_index=True)
#
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
#             # Load the image and resize to (224, 224) for ResNet50
#             img = load_img(image_path, target_size=(224, 224))
#             img_array = img_to_array(img)
#             img_preprocessed = preprocess_input(img_array)  # Preprocess for ResNet50
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
# # # Example: Load your data (replace this with your actual data loading code)
# # # For demonstration, we'll create some dummy data
# # train_images = np.random.rand(1000, 224, 224, 3)  # 1000 training images
# # train_labels = np.random.randint(0, 2, size=(1000, 2))  # 1000 training labels (one-hot encoded)
# # valid_images = np.random.rand(200, 224, 224, 3)  # 200 validation images
# # valid_labels = np.random.randint(0, 2, size=(200, 2))  # 200 validation labels (one-hot encoded)
#
# # Define ResNet50 model
# def create_resnet50_model(input_shape, num_classes):
#     # Load ResNet50 base model with pre-trained weights
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
#
#     # Freeze the base model layers for transfer learning
#     for layer in base_model.layers:
#         layer.trainable = False
#
#     # Add custom classification layers
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(256, activation='relu')(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
#
#     # Compile the model
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer=Adam(learning_rate=1e-4),
#                   loss=tf.losses.CategoricalCrossentropy(),
#                   metrics=['accuracy'])
#     return model
#
# def lr_schedule(epoch, lr):
#     # Example: Decrease learning rate by 10% every 5 epochs
#     if epoch % 5 == 0 and epoch > 0:
#         lr *= 0.9
#     return float(lr)  # Ensure the return value is a float
#
# # Create LearningRateScheduler callback with the corrected schedule function
# learning_rate_scheduler = LearningRateScheduler(lr_schedule)
#
# # Callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
#
# # Model configuration
# input_shape = (224, 224, 3)
# num_classes = 2  # Yes and No classesn
# model = create_resnet50_model(input_shape, num_classes)
#
# # Training the model
# history = model.fit(
#     train_images,
#     train_labels,
#     batch_size=32,
#     epochs=50,
#     validation_data=(valid_images, valid_labels),
#     callbacks=[learning_rate_scheduler]
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
#

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split

import warnings
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

## 修改 数据泄露 问题
train_data_yes, temp_data_yes = train_test_split(data_yes, test_size=0.3, shuffle=False)
valid_data_yes, test_data_yes = train_test_split(temp_data_yes, test_size=0.5, shuffle=False)

train_data_no, temp_data_no = train_test_split(data_no, test_size=0.3, shuffle=False)
valid_data_no, test_data_no = train_test_split(temp_data_no, test_size=0.5, shuffle=False)

train_data = pd.concat([train_data_yes, train_data_no], ignore_index=True)
valid_data = pd.concat([valid_data_yes, valid_data_no], ignore_index=True)
test_data = pd.concat([test_data_yes, test_data_no], ignore_index=True)

# Function to preprocess images and create data lists
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
            try:
                # Load the image and resize to (224, 224) for ResNet50
                img = load_img(image_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_preprocessed = preprocess_input(img_array)  # Preprocess for ResNet50

                images.append(img_preprocessed)
                labels.append(label)
            except Exception as e:
                print(f"Skipping corrupted/invalid image: {image_path}")
                continue

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

# Define data augmentation for the training set
data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define ResNet50 model
def create_resnet50_model(input_shape, num_classes):
    # Load ResNet50 base model with pre-trained weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model layers for transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Compile the model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=tf.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

def lr_schedule(epoch, lr):
    # Example: Decrease learning rate by 10% every 5 epochs
    if epoch % 5 == 0 and epoch > 0:
        lr *= 0.9
    return float(lr)  # Ensure the return value is a float

# Create LearningRateScheduler callback with the corrected schedule function
learning_rate_scheduler = LearningRateScheduler(lr_schedule)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Model configuration
input_shape = (224, 224, 3)
num_classes = 2  # Yes and No classes
model = create_resnet50_model(input_shape, num_classes)

# Training the model
history = model.fit(
    train_images,
    train_labels,
    batch_size=32,
    epochs=15,
    validation_data=(valid_images, valid_labels),
    callbacks=[learning_rate_scheduler]
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