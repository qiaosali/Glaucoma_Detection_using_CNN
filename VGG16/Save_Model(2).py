from tensorflow.keras.applications import VGG16

# Save the VGG16 model and weights
vgg16_weights_path = 'F:/Thesis/Project_PyQt6/pyqt6/VGG16 Model/vgg16_model_weights.weights.h5'
vgg16_model_path = 'F:/Thesis/Project_PyQt6/pyqt6/VGG16 Model/vgg16_model.h5'

# Instantiate the VGG16 model
vgg16_model = VGG16(weights='imagenet')  # Load with pre-trained ImageNet weights

# Save the model weights and the model
vgg16_model.save_weights(vgg16_weights_path)
vgg16_model.save(vgg16_model_path)

print("VGG16 model and weights saved successfully!")
