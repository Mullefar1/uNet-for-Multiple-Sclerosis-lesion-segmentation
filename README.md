# uNet-for-Multiple-Sclerosis-lesion-segmentation
uNet based segmentation for Multiple Sclerosis lesions in python using Keras. 

The segmentation process is split in two. Modelfitting.py are training the model and saving the model used for predicting on test data and Predict.py is predicting on the test data

## **uNet structure:**

![u-net-architecture](https://user-images.githubusercontent.com/56428296/135154140-62d2df12-e84d-4502-9970-5f1dfc8abe31.png)

## **Data:**

![GT](https://user-images.githubusercontent.com/56428296/135148011-aa05fe06-50b4-43e3-a9f9-cc4d612c614d.PNG)

## **Results:**

![result](https://user-images.githubusercontent.com/56428296/135148021-978b5322-0b54-430f-a61e-7e39d0c39788.PNG)

## **OBS:**

The Modelfitting.py uses early stopping. Make sure to change this parameter depended on your purpose. 

Furthermore, the code contains different loss functions and metrics. These can be changes accordingly and based on purpose. The optimizers can be found here: https://keras.io/api/optimizers/   

The uNet should be suitable for other segmentations approaches such as brain tumor segmentation, cerebral hemorrhage segmentation etc.

## **Problems or Questions?:**

Just send an Issue and I will look into it 

## **Future:**

Currently working on different supervised methods for segmentation. The idea is to develop a good enough supervised segmentation of Multiple Sclerosis lesions and then use the supervised segmentation as input to the uNet for complete automatic segmentation (no manual segmentation is needed). Stay tuned for more!


