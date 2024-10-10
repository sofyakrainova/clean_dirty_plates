import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 20
TARGET_SIZE = 250
TRAINING_DIR = "../../Python scripts/Kaggle_data/plates/train"
VALIDATION_DIR = "../../Python scripts/Kaggle_data/plates/validation"
EPOCHS = 100


def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators

  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images

  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  train_datagen = ImageDataGenerator( rescale = 1.0/255.,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode=('nearest'),
                                      )
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=BATCH_SIZE,
                                                      class_mode="binary",
                                                      target_size=(TARGET_SIZE, TARGET_SIZE),
                                                      color_mode="rgb",
                                                      shuffle=True,
                                                      )

  validation_datagen = ImageDataGenerator( rescale = 1.0/255. )
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=8,
                                                                class_mode="binary",
                                                                target_size=(TARGET_SIZE, TARGET_SIZE),
                                                                color_mode="rgb",
                                                                )
  return train_generator, validation_generator

train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

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
                     overwrite=True,
                     directory="tuner_dir",
                     project_name="plates"
                     )
tuner.search(train_generator,
             epochs=EPOCHS,
             validation_data = validation_generator
             )

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete.\n
 The optimal filters1 is {best_hps.get('filters1')}. \n 
 The optimal kernel1 is {best_hps.get('kernel1')}. \n 
 The optimal filters2 is {best_hps.get('filters2')}. \n 
 The optimal kernel2 is {best_hps.get('kernel2')}. \n 
 The optimal filters3 is {best_hps.get('filters3')}. \n 
 The optimal kernel3 is {best_hps.get('kernel3')}. \n 
 The optimal learning rate is {best_hps.get('learning_rate')}. \n
 Number units {best_hps.get('units')} \n
 """)