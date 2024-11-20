import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image

def load_data_from_coco(folder_path):
    with open(os.path.join(folder_path, "_annotations.coco.json"), 'r') as f:
        annotations = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    images = annotations['images']
    annotations_data = annotations['annotations']
    
    image_label_map = {}
    for ann in annotations_data:
        image_id = ann['image_id']
        label = categories[ann['category_id']]
        image_name = next(img['file_name'] for img in images if img['id'] == image_id)
        image_label_map[os.path.join(folder_path, image_name)] = label
    
    return image_label_map

# To download dataset: https://www.kaggle.com/datasets/ammarnassanalhajali/bccd-coco
def prepare_dataset(base_path):
    datasets = {}
    for folder in ['train', 'valid', 'test']:
        folder_path = os.path.join(base_path, folder)
        datasets[folder] = load_data_from_coco(folder_path)
    return datasets

base_path = r"bbcd"
datasets = prepare_dataset(base_path)

def load_images_and_labels(dataset):
    images, labels = [], []
    class_map = {}
    for i, (file_path, label) in enumerate(dataset.items()):
        img = Image.open(file_path).resize((128, 128))  
        images.append(np.array(img) / 255.0)
        if label not in class_map:
            class_map[label] = len(class_map)
        labels.append(class_map[label])
    return np.array(images), np.array(labels), class_map

train_images, train_labels, class_map = load_images_and_labels(datasets['train'])
valid_images, valid_labels, _ = load_images_and_labels(datasets['valid'])
test_images, test_labels, _ = load_images_and_labels(datasets['test'])

print("Number of training images:", len(train_images))
print("Number of validation images:", len(valid_images))
print("Number of testing images:", len(test_images))

def display_samples(images, labels, class_map):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        idx = np.random.randint(len(images))
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[idx])
        plt.title(list(class_map.keys())[list(class_map.values()).index(labels[idx])])
        plt.axis("off")
    plt.show()

display_samples(train_images, train_labels, class_map)

def augment_data_on_the_fly(train_images, train_labels):
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        rescale=1./255 
    )
    datagen_flow = datagen.flow(train_images, train_labels, batch_size=32)
    return datagen_flow

def build_and_train_model(X_train, y_train, X_valid, y_valid, class_map, epochs=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_map), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs)
    return model, history

print("\nTraining without augmentation...")
model_no_aug, history_no_aug = build_and_train_model(train_images, train_labels, valid_images, valid_labels, class_map)

train_datagen_flow = augment_data_on_the_fly(train_images, train_labels)

print("\nTraining with augmentation...")
history_aug = model_no_aug.fit(train_datagen_flow, validation_data=(valid_images, valid_labels), epochs=10)

print("\nAccuracy without augmentation:", history_no_aug.history['val_accuracy'][-1])
print("Accuracy with augmentation:", history_aug.history['val_accuracy'][-1])

def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

print("\nPlotting history for model without augmentation...")
plot_history(history_no_aug)

print("\nPlotting history for model with augmentation...")
plot_history(history_aug)



