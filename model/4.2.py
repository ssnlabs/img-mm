import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, accuracy_score

dataset_dir = r"horse-or-human"  
train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary',
    validation_split=0.3,
    subset='training',
    seed=42
)

val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary',
    validation_split=0.3,
    subset='validation',
    seed=42
)
for images, labels in train_data.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):  
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i])
        plt.axis("off")
    plt.show()

normalization_layer = layers.Rescaling(1.0 / 255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=3)

train_loss, train_acc = model.evaluate(train_data)
print(f"\nTraining Accuracy: {train_acc:.2f}")

val_loss, val_acc = model.evaluate(val_data)
print(f"\nValidation (Testing) Accuracy: {val_acc:.2f}")

# y_pred = np.round(model.predict(val_data).flatten())
# y_true = np.concatenate([label.numpy() for _, label in val_data], axis=0)

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Horse', 'Human'], yticklabels=['Horse', 'Human'])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()
