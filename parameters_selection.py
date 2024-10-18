import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import numpy as np
import os
import pandas as pd

BATCH_SIZE = 20
TARGET_SIZE = 250
EPOCHS = 100

all_train = "../../Python scripts/Kaggle_data/plates/all_train"
filenames = os.listdir(all_train)

image_paths = np.array([os.path.join(all_train, f) for f in filenames if os.path.isfile(os.path.join(all_train, f))])
labels = np.array(["dirty", "clean"]*20)

all_train = "../../Python scripts/Kaggle_data/plates/all_train"
filenames = os.listdir(all_train)

image_paths = np.array([os.path.join(all_train, f) for f in filenames if os.path.isfile(os.path.join(all_train, f))])
labels = np.array(["dirty", "clean"]*20)

# Number of folds for cross-validation
k_folds = 10

# Data augmentation setup using ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1.0/255)

validation_gen = ImageDataGenerator(rescale=1.0/255)

def prepare_df(paths, labels, idx):
    return pd.DataFrame({'filename': paths[idx], 'class': labels[idx]})

def model_builder(hp):
  model = tf.keras.Sequential()
  filters1 = hp.Int('filters1', min_value=8, max_value=32, step=2)
  kernel1 = hp.Int('kernel1', min_value=2, max_value=5, step=1)
  model.add(tf.keras.layers.Conv2D(filters1,
                                   (kernel1, kernel1),
                                   activation='relu',
                                   input_shape=(TARGET_SIZE, TARGET_SIZE, 3)),)
  model.add(tf.keras.layers.MaxPooling2D(2,2))
  filters2 = hp.Int('filters2', min_value=8, max_value=32, step=2)
  kernel2 = hp.Int('kernel2', min_value=2, max_value=5, step=1)
  model.add(tf.keras.layers.Conv2D(filters2,
                                   (kernel2,kernel2),
                                   activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(2,2),)
  filters3 = hp.Int('filters3', min_value=8, max_value=64, step=4)
  kernel3 = hp.Int('kernel3', min_value=2, max_value=5, step=1)
  model.add(tf.keras.layers.Conv2D(filters3, (kernel3,kernel3), activation='relu'),)
  model.add(tf.keras.layers.MaxPooling2D(2,2),)
  model.add(tf.keras.layers.Flatten())
  units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(tf.keras.layers.Dense(units, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  # Tune the learning rate for the optimizer
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
  model.compile(optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"])

  return model

tuner = kt.Hyperband(model_builder,
                     max_epochs=20,
                     objective="val_accuracy",
                     #overwrite=True,
                     directory="tuner_dir",
                     project_name="plates"
                     )

tkf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_num = 1
all_fold_scores = []
params_list = []

for fold_idx, (train_idx, val_idx) in enumerate(tkf.split(image_paths)):
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
    tuner = kt.Hyperband(model_builder,
                         max_epochs=50,
                         objective="val_accuracy",
                         overwrite=True,
                         directory="tuner_dir",
                         project_name="plates"
                         )
    # Use tuner to search hyperparameters for this fold
    tuner.search(train_generator,
                 epochs=EPOCHS,
                 validation_data=val_generator,
                 verbose=1)

    # Get the best hyperparameters for this fold
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    params_list.append(best_hps)
    # Build the best model
    model = tuner.hypermodel.build(best_hps)

    # Train the best model on the fold data
    history = model.fit(train_generator, epochs=10, validation_data=train_generator, verbose=0)

    # Evaluate the model on validation data
    val_accuracy = model.evaluate(train_generator, verbose=0)[1]  # Accuracy is at index 1
    print(f"Fold {fold_num} Validation Accuracy: {val_accuracy}")

    # Store the validation accuracy for this fold
    all_fold_scores.append(val_accuracy)

    fold_num += 1

# 4. Calculate the average accuracy across all folds
average_accuracy = np.mean(all_fold_scores)
print(f"Average Validation Accuracy across {tkf.get_n_splits()} folds: {average_accuracy}")

for param in params_list:
    print(f"""
    The hyperparameter search is complete.\n
     The optimal learning rate is {param.get('learning_rate')}. \n
     Number filters1 {param.get('filters1')}, kernel1 {param.get('kernel1')} \n
     Number filters2 {param.get('filters2')}, kernel2 {param.get('kernel2')} \n
     Number filters3 {param.get('filters3')}, kernel3 {param.get('kernel3')} \n
     Number units {param.get('units')} \n 
     """)
