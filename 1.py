#1
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'image.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
smoothed_image = cv2.GaussianBlur(image, (31, 31), 0)

grad_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(grad_x, grad_y)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(2, 2, 2), plt.imshow(smoothed_image, cmap='gray'), plt.title('Smoothed Image (Gaussian Filter)'), plt.axis('off')
plt.subplot(2, 2, 3), plt.imshow(grad_x, cmap='gray'), plt.title('Gradient in X-direction'), plt.axis('off')
plt.subplot(2, 2, 4), plt.imshow(grad_y, cmap='gray'), plt.title('Gradient in Y-direction'), plt.axis('off')
plt.tight_layout()

plt.figure(figsize=(6, 6))
plt.imshow(magnitude, cmap='hot')
plt.title('Edge Magnitude (Sobel Operator)')
plt.axis('off')
plt.show()

#2
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

(ds_train, ds_test), ds_info = tfds.load('horses_or_humans', split=['train', 'test'], with_info=True, as_supervised=True)

train_size = ds_info.splits['train'].num_examples
test_size = ds_info.splits['test'].num_examples
print(f"Number of training images: {train_size}")
print(f"Number of testing images: {test_size}")

def plot_samples(dataset, num_images=9):
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(dataset.take(num_images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Horse" if label == 0 else "Human")
    plt.show()

plot_samples(ds_train)

def preprocess_data(dataset, batch_size=32):
    def normalize_image(image, label):
        image = tf.image.resize(image, (128, 128))  
        image = tf.cast(image, tf.float32) / 255.0  
        return image, label
    return dataset.map(normalize_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = preprocess_data(ds_train)
test_ds = preprocess_data(ds_test)

base_model = tf.keras.applications.ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=5, validation_data=test_ds)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc}")