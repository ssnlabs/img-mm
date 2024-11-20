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
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_no_aug = model.fit(train_ds, epochs=5, validation_data=test_ds)

plt.plot(history_no_aug.history['accuracy'], label='Training Accuracy')
plt.plot(history_no_aug.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Before Augmentation')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy before augmentation: {test_acc}")

def augment_data(dataset):
    def augment(image, label):
        image = tf.image.resize(image, (128, 128))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label

    return dataset.map(augment).batch(32).prefetch(tf.data.AUTOTUNE)

augmented_train_ds = augment_data(ds_train)
model_aug = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_aug.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_aug = model_aug.fit(augmented_train_ds, epochs=5, validation_data=test_ds)

plt.plot(history_aug.history['accuracy'], label='Training Accuracy')
plt.plot(history_aug.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy After Augmentation')
plt.legend()
plt.show()

test_loss_aug, test_acc_aug = model_aug.evaluate(test_ds)
print(f"Test accuracy after augmentation: {test_acc_aug}")
print(f"Test accuracy before augmentation: {test_acc}")
print(f"Test accuracy after augmentation: {test_acc_aug}")

labels = ['Before Augmentation', 'After Augmentation']
accuracy_values = [test_acc, test_acc_aug]

plt.bar(labels, accuracy_values, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy Before and After Augmentation')
plt.ylim([0, 1])
plt.show()



