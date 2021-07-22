# AI-Road-Semantic-Segmentation

## Introduction

<p align="center">
<img src="https://github.com/PatelVatsalB21/AI-Road-Segmentation/blob/test/model/performance/Semantic%20Segmantation%20Demo.gif"/>
</p>

Image segmentation is a computer vision task in which we label specific regions of an image according to what's being shown. The goal of Semantic Image Segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we're predicting for every pixel in the image, this task is commonly referred to as dense prediction. We only care about the category of each pixel i.e. if you have two objects of the same category in your input image, the segmentation map does not inherently distinguish these as separate objects.


## Dataset

The dataset used for Semantic Segmantation of Road is available on Kaggle. It is a free and open-source dataset with 10000 images taken from CARLA Simulator. The dataset is divided in two parts of 5000 images each, one for RGB images and other for Segmentated images(output). The segmentated images are pixel labelled images and have 13 categories each for different objects(e.g. cars, buildings, road). The size of whole dataset is around 5GB and available to download from [here](https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge).


## Model Overview

The basic structure of semantic segmentation models is based on the U-Net model. U-Net structure is as shown in the image below:

<p align="center">
<img src="https://github.com/PatelVatsalB21/AI-Road-Segmentation/blob/test/model/performance/U-Net%20Structure.png"/>
</p>

The left side of the model represents any feature extraction network trained for image classification. Once those features are extracted they are then further processed at different scales. The reason for this is two-fold. Firstly, your model will very likely encounter objects of many different sizes; processing the features at different scales will give the network the capacity to handle those different sizes. Second, when performing segmentation there is a tradeoff. If you want good classification accuracy, then you’ll definitely want to process those high level features from later in the network since they are more discriminative and contain more useful semantic information. On the other hand, if you only process those deep features, you won’t get good localisation because of the low resolution.


## Model Structure 

The internal layer structer of model used in this project is shown below.

<p align="center">
<img src="https://github.com/PatelVatsalB21/AI-Road-Segmentation/blob/test/model/performance/Model_Structure.png"/>
</p>
 
Model follows specific pattern after the Input layer with multiple groups of Convolution, Batch Normalization and Activation layers throughout the structure. After the first Convolution(Conv2D) layer the structure divides on two separate flows. The reason for this is as said above, we want to focus on both the main features and residual(parts of object except features) so that it can localize well. After each such processing layes are added and this is done four times in the model. After the second addition layer(add_2) the size of output image is (16,16,256). Hence after feature extraction from the small parts of image, one flow goes throught same feature extraction while other is upsampled and used for residual extraction. Finally after sixth addition layer(add_6) there is a convolution layer. This Convolution layer has 13 different outputs with Softmax activation for multi channel segmantation of the image. Its output is of shape (x, 256, 256, 13). Here x is number of input images while the following two are output dimensions and the last is for channels with each channel for different objects(e.g. cars, road, vegetations, buildings etc).


## Model Training 

The model is trained from mentioned dataset. The parameters used in model training are:
- ```optimizer="rmsprop"```
- ```loss="sparse_categorical_crossentropy"```
- ```callbacks = [ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-12)]```
- ```batch_size=32```
- ```epochs=40```
- ```validation_split=0.1```

The stats of training are as shown in images below:

<img src="https://github.com/PatelVatsalB21/AI-Road-Segmentation/blob/test/model/performance/Accuracy.png"/>

<img src="https://github.com/PatelVatsalB21/AI-Road-Segmentation/blob/test/model/performance/Loss.png"/>


## Predictions

The model achieved 91.92% validation accuracy and 93.8% test accuracy and with test loss and validation loss below 0.25 . A sample of the preditions by model is as shown below.

<img src="https://github.com/PatelVatsalB21/AI-Road-Segmentation/blob/test/model/performance/Truth.png"/>

<img src="https://github.com/PatelVatsalB21/AI-Road-Segmentation/blob/test/model/performance/Prediction.png"/>

<img src="https://github.com/PatelVatsalB21/AI-Road-Segmentation/blob/test/model/performance/Colour_Coded_Prediction.png"/>

To download the pre-trained model used in the project [click here](https://github.com/PatelVatsalB21/AI-Road-Segmentation/raw/test/model/model/semantic_model(91.92).h5)


## Code & Issues
If you are a developer and you wish to contribute to the project please fork the project
and submit a pull request to **test** branch.
Follow [Github Flow](https://help.github.com/articles/github-flow/) for collaboration!
