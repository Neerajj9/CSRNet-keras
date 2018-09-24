# CSRNet-keras
Implementation of the CRNet paper (CVPR 18) in keras-tensorflow

### CVPR 2018 Paper : https://arxiv.org/abs/1802.10062

### Official Pytorch Implementation : https://github.com/leeyeehoo/CSRNet-pytorch

As we were searching on the internet, we could not find any keras implementation of the state of the art CSRNet paper. A large part of the deep learning community uses keras-tensorflow to implement their neural network models. Thus, we thought of implementing the CSRNet model in keras-tensorflow. Another perk of implementing it in keras is that it can be easily deployed on an android system due to tensorflow support.

# Data Preprocessing  :
The main objective in data preprocessing was to convert the ground truth provide by the ShanghaiTech dataset into density maps. For a given image the dataset provided a sparse matrix consisting of the coordinates where a person is present in that image. This sparse matrix was converted into a 2D density map ny passing through a Gaussian Filter. The sum of all the cells in the density map results in the actual count of people in that particular image. Refer the `Preprocess.ipynb` notebook for the same.
