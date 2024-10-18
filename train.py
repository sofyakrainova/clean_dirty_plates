import keras.src.layers
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

TARGET_SIZE = 250
EPOCHS = 100
all_train = "../../Python scripts/Kaggle_data/plates/all_train"
filenames = os.listdir(all_train)

image_paths = np.array([os.path.join(all_train, f) for f in filenames if os.path.isfile(os.path.join(all_train, f))])
labels = np.array(["dirty", "clean"]*20)

# Number of folds for cross-validation
k_folds = 3

# Data augmentation setup using ImageDataGenerator
train_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_gen = ImageDataGenerator(rescale=1.0/255)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3)),
        tf.keras.layers.Conv2D(28, (2, 2), activation='relu', ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(24, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(20, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(320, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def prepare_df(paths, labels, idx):
    return pd.DataFrame({'filename': paths[idx], 'class': labels[idx]})

# Initialize KFold cross-validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42, )

# Array to store accuracy for each fold
fold_accuracies = []

# Iterate over each fold
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"Training fold {fold_idx + 1}/{k_folds}")
    # Split data into training and validation sets based on indices
    train_df = prepare_df(image_paths, labels, train_idx)
    valid_df = prepare_df(image_paths, labels, val_idx)

    # Create train and validation generators
    train_generator = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        folder = all_train,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = validation_gen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='filename',
        y_col='class',
        folder=all_train,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=32,
        class_mode='binary'
    )

    # Create a fresh model for each fold
    model = create_model()

    # Train the model on this fold
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30
    )

    # Evaluate the model on the validation fold
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Fold {fold_idx + 1} - Validation Accuracy: {val_accuracy:.4f}")

    # Store the accuracy for this fold
    fold_accuracies.append(val_accuracy)

# Compute the average accuracy across all folds
average_accuracy = np.mean(fold_accuracies)
print(f"Average Accuracy across {k_folds} folds: {average_accuracy:.4f}")

all_df = prepare_df(image_paths, labels, np.arange(0, len(image_paths)))
train_generator = train_gen.flow_from_dataframe(
        dataframe=all_df,
        x_col='filename',
        y_col='class',
        folder = all_train,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=32,
        class_mode='binary'
    )

model.fit(
        train_generator,
        epochs=100)
results = model.evaluate(train_generator)
print("test loss, test acc:", results)

# Save the weights
model.save('trained_model.keras')
