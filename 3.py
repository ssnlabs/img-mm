import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

image = cv2.imread(r'image.jpg', cv2.IMREAD_GRAYSCALE)

radius = 5
n_points = 8 * radius

lbp_image = local_binary_pattern(image, n_points, radius, method="uniform")

lbp_image_normalized = np.uint8(255 * (lbp_image - lbp_image.min()) / (lbp_image.max() - lbp_image.min()))

height, width = lbp_image.shape
lbp_flat = lbp_image_normalized.reshape(-1, 1)

num_clusters = 4 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(lbp_flat)

segmented_image = labels.reshape(height, width)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(image, cmap="gray")
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(lbp_image_normalized, cmap="gray")
axs[1].set_title("LBP Image")
axs[1].axis("off")

axs[2].imshow(segmented_image, cmap="viridis")
axs[2].set_title("Segmented Image (Texture Regions)")
axs[2].axis("off")

plt.tight_layout()
plt.show()

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


