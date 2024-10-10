import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

TARGET_SIZE = 250
EPOCHS = 100
TRAINING_DIR = "../../Python scripts/Kaggle_data/plates/train"
VALIDATION_DIR = "../../Python scripts/Kaggle_data/plates/validation"
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
  train_datagen = ImageDataGenerator( rescale = 1.0/255,
                                      rotation_range=180,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      vertical_flip = True,
                                      fill_mode=('nearest'),
                                      )
  train_generator = train_datagen.flow_from_directory(
                                                      directory=TRAINING_DIR,
                                                      batch_size=20,
                                                      class_mode="binary",
                                                      target_size=(TARGET_SIZE, TARGET_SIZE),
                                                      color_mode="rgb",
                                                      shuffle=True,
                                                      )

  validation_datagen = ImageDataGenerator( rescale = 1.0/255. )
  validation_generator = validation_datagen.flow_from_directory(
                                                                directory=VALIDATION_DIR,
                                                                batch_size=8,
                                                                class_mode="binary",
                                                                target_size=(TARGET_SIZE, TARGET_SIZE),
                                                                color_mode="rgb",
                                                                )
  return train_generator, validation_generator

train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)
print(validation_generator.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(12, (2,2), activation='relu', input_shape=(TARGET_SIZE, TARGET_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(26, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(20, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(480, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(learning_rate=1e-04),
                loss="binary_crossentropy",
                metrics=["accuracy"])



history = model.fit(train_generator,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=validation_generator)

model.fit(validation_generator,
                    epochs=20,
                    verbose=1,
                    )


# Plot utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.grid()
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

# Plot the accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# Save the weights
model.save('trained_model.keras')