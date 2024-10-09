# clean_dirty_plates

Data for this project was taken from this Kaggle competition https://www.kaggle.com/competitions/platesv2/data  
Tricky part is than the amount of train data is very small: 19 clean and 19 dirty images.   

====================== Transfer learning model ===========================
Here we will retrain existing model for image classification Inception V3. Weights for the model can be downloaded from here:
https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5  
We added 2 dense layers, after parameters tuning hwe have layer1 736 units, layer2 128 units, learning rate 1e-5.
Since we have small amout of data, we will finalize training with validation data. After 100 epochs we have:    
Train accuracy  
Validation accuracy  
Test accuracy 0.858  
  
After we added image augmentation we have:  
Train accuracy:  
Validation accuracy:  
Test accuracy:  
