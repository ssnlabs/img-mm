import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Load the BCCD dataset from TensorFlow Datasets
dataset, info = tfds.load('blood_cells', with_info=True, as_supervised=True)

# The dataset is split into 'train' and 'test' by default
train_data = dataset['train']
val_data = dataset['test']

# Check the number of training and testing images
train_size = info.splits['train'].num_examples
val_size = info.splits['test'].num_examples

print(f'Training Images: {train_size}')
print(f'Validation Images: {val_size}')

# Plot a few images from the dataset
def plot_images(dataset, num_images=5):
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(dataset.take(num_images)):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()

plot_images(train_data)

# Image Augmentation function
def augment_image(image):
    image = tf.image.random_flip_left_right(image)  # Random horizontal flip
    image = tf.image.random_flip_up_down(image)    # Random vertical flip
    image = tf.image.random_contrast(image, lower=0.2, upper=0.5)  # Random contrast
    return image

# Apply augmentation on the dataset
train_data = train_data.map(lambda image, label: (augment_image(image), label))
val_data = val_data.map(lambda image, label: (augment_image(image), label))

# Normalize the images
normalization_layer = layers.Rescaling(1.0 / 255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# Build a CNN model
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(128, 128, 3)),  # Adjust size to 128x128 for simplicity
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification, change to softmax for multi-class
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data.batch(32), validation_data=val_data.batch(32), epochs=10)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
