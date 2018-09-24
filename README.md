# CSRNet-keras
Implementation of the CRNet paper (CVPR 18) in keras-tensorflow

### CVPR 2018 Paper : https://arxiv.org/abs/1802.10062

### Official Pytorch Implementation : https://github.com/leeyeehoo/CSRNet-pytorch

As we were searching on the internet, we could not find any keras implementation of the state of the art CSRNet paper. A large part of the deep learning community uses keras-tensorflow to implement their neural network models. Thus, we thought of implementing the CSRNet model in keras-tensorflow. Another perk of implementing it in keras is that it can be easily deployed on an android system due to tensorflow support.

# Dataset :
The dataset used is ShanghaiTech dataset available here : [Drive Link](https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view)

The dataset is divided into two parts, A and B. The A part consists of images with a high density of crowd. The B part consists of images with images of sparse crowd scenes.   

# Data Preprocessing  :
The main objective in data preprocessing was to convert the ground truth provide by the ShanghaiTech dataset into density maps. For a given image the dataset provided a sparse matrix consisting of the coordinates where a person is present in that image. This sparse matrix was converted into a 2D density map ny passing through a Gaussian Filter. The sum of all the cells in the density map results in the actual count of people in that particular image. Refer the `Preprocess.ipynb` notebook for the same.

# Model :
The CSRNet model uses Convolutional Neural Networks to map the input image to it's respective density map. The model does not make use of any fully connected layers and thus the size of the input image is variable. As a result, the model learns from a large amount of varied data and there is no information loss considering the image resolution. There is no need of reshaping/resizing the image while inferencing. The model architecture is such that considering the input image to be (x,y,3), the output is a desnity map of size (x/8,y/8,1).

The model architecture is divide into two parts, front-end and back-end. The front-end consists of 13 pretrained layers of the VGG16 model ( 10 Convolution layers and 3 MaxPooling layers ). The fully connected layers of the VGG16 are not taken. The back-end comprises of Dilated Convolution layers. The dilation rate at which maximum accuracy was obtained was experimentally found out be 2 as suggested in the CSRNet paper.

Batch Normalisation functionality is also provided in the code. As VGG16 does not have any BN layers, we build a custom VGG16 model and ported pretrained weights of VGG16 to this model.

In keras it is difficult to train a model where the size of the input image is variable. Keras does not allow variable size inputs to be trained in the same batch of inputs. One way to tackle this is to combine all images having the same image dimension and train them as a batch. The ShanghaiTech dataset does not contain many images having the same image dimension. Another approach is to train each image independantly and run a loop over all images. This approach is not efficient in terms of memory usage, computations and time. Thus, we built a custom data generator in keras to efficiently train variable sized images. With a data generator, efficient memory usage takes place and the time taken for training reduces drastically.

The paper also specifies cropping of images as a part of data augmentation. However, the Pytorch implementation does not use cropping of images while training. Hence we have provided a function `preprocess_input()` which can be used inside `image_generator()` to add the cropping functionality.

