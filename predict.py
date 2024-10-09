from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

TEST_DIR = "../../Python scripts/Kaggle_data/plates/"
TARGET_SIZE = 250
savedModel=load_model("trained_model.keras")
print(savedModel.summary())

test_datagen = ImageDataGenerator(rescale=1/255.)

test_generator = test_datagen.flow_from_directory(TEST_DIR,
                              classes=['test'],
                              class_mode=None,
                              shuffle=False,
                              color_mode="rgb",
                              target_size=(TARGET_SIZE, TARGET_SIZE),
                              batch_size= 1)

preds = savedModel.predict(test_generator).flatten()
preds = np.array(["dirty" if label>=0.5 else "cleaned" for label in preds])
filenames=test_generator.filenames

results=pd.DataFrame({"id":filenames,
                      "label":preds})
results["id"] = results["id"].apply(lambda x: x[5:-4])
results.to_csv("submission.csv",index=False)