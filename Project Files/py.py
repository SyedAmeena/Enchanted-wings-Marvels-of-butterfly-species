import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report

# Step 1: Set data path
DATA_DIR = "butterfly_dataset"  # Folder should have subfolders for each species

# Step 2: Image preprocessing
image_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = image_gen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = image_gen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Step 3: Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# Step 6: Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Butterfly Species Classification Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Optional: Evaluate on validation data
val_images, val_labels = next(val_data)
predictions = model.predict(val_images)
print(classification_report(np.argmax(val_labels, axis=1), np.argmax(predictions, axis=1)))