# Data Augmentation, Depthwise Separable Convolution and Dilated Convolution

## 1. Data Augmentation:
Data augmentation is a technique used in machine learning to increase the amount of training data available to a model. It involves generating new training examples by applying transformations to existing data. Common transformations include changing the brightness, rotation, and scaling of images, adding noise, and flipping. Data augmentation can help improve model performance by reducing overfitting and increasing generalizability. It can also be used to balance class distributions in datasets. There are open source libraries and pre-built models available that can implement data augmentation for various applications such as image classification and object detection.  

We have utilized the **Albumentations library** in this code. 
1. A.HorizontalFlip(p=1),
2. A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
3. A.CoarseDropout(max_holes=1, min_holes=1, max_height=16, min_height=16, max_width=16, min_width=16, fill_value=0, mask_fill_value=None).

## 2.  Depthwise Separable Convolution:
Depthwise separable convolution is a type of convolution operation used in deep learning for image and video processing. It is a computationally more efficient method for performing convolutions, as it separates the depthwise and pointwise operations, which reduces the number of parameters and computations required. Depthwise separable convolution has shown to be effective in several computer vision tasks, such as object detection, image classification, and segmentation. The concept was introduced in a research paper bypenzdifford81 is widely used in mobile devices and other resource-constrained computing environments.  

## 3.  Dilated Convolution:
Dilated convolution is a mathematical operation used in image and signal processing. It is a type of convolution that increases the spatial separation between the filters used to apply the operation. This allows the operation to capture more complex features from the input signal. Dilated convolution is often used in computer vision tasks such as object detection and segmentation. Finally, Dilated convolution can be defined as function that takes an image as input and produces an output image, where at each output pixel, the corresponding output value is computed as a weighted sum of the values in the input pixel’s neighbourhood, with the weights determined by a learnable filter.  

## 4. Target:
1. Targeted Accuracy: 85%
2. Parameters: 200k
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5.  Data Augmentation using albumentations library: <br>
A. A.HorizontalFlip(p=1).<br>
B. A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5).<br>
C. A.CoarseDropout(max_holes=1, min_holes=1, max_height=16, min_height=16, max_width=16, min_width=16, fill_value=(mean of your dataset), mask_fill_value=None).


## 5. Analysis:
