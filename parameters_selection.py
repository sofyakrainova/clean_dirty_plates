import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

BATCH_SIZE = 20
TARGET_SIZE = 250
TRAINING_DIR = "../../Python scripts/Kaggle_data/plates/train"
VALIDATION_DIR = "../../Python scripts/Kaggle_data/plates/validation"
EPOCHS = 100
local_weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


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

pre_trained_model = InceptionV3(input_shape = (TARGET_SIZE, TARGET_SIZE, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)
# Making sure trained layers will stay intact
for layer in pre_trained_model.layers:
  layer.trainable = False

# Last trained layer we will use
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

def model_builder(hp):
    x = tf.keras.layers.Flatten()(last_output)
    units1 = hp.Int('units1', min_value=32, max_value=1024, step=32)
    x = tf.keras.layers.Dense(units1, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    units2 = hp.Int('units2', min_value=32, max_value=1024, step=32)
    x = tf.keras.layers.Dense(units2, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Append the dense network to the base model
    model = tf.keras.Model(pre_trained_model.input, x)

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
 The optimal learning rate is {best_hps.get('learning_rate')}. \n
 Number units1 {best_hps.get('units1')} \n
 Number units2 {best_hps.get('units2')} \n 
 """)