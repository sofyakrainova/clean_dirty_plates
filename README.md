# clean_dirty_plates

Data for this project was taken from this Kaggle competition https://www.kaggle.com/competitions/platesv2/data  
Tricky part is than the amount of train data is very small: 19 clean and 19 dirty images.   

====================== Transfer learning model ===========================
Here we will retrain existing model for image classification Inception V3. Weights for the model can be downloaded from here:
https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5  
We added 2 dense layers, after parameters tuning hwe have layer1 736 units, layer2 128 units, learning rate 1e-5.
Since we have small amout of data, we will finalize training with validation data. After 100 epochs we have:    
Train accuracy  1.00  
Validation accuracy 0.75   
Test accuracy 0.858  
  
After we added image augmentation we have:  
Train accuracy:  1.00  
Validation accuracy: 0.75  
Test accuracy: 0.879  
========================= Custom model =======================================
Custom model with CNN does not perform good, so I will not post it alone. Using K-fold parameter tunning we have these params:
filters1 28, kernels1 2, filters2 24, kernels2 3, filters3 20, kernels3 3, units 320, learninng rate 0.01   
performance achieved so far after 100 epochs:  
Train 0.697  
Validation (across 3 k-folds): 0.376  
Test 0.653    
