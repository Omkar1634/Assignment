# Handwritten Digit Recognition using Convolutional Neural Networks with Pytorch

The MNIST dataset comprises 70,000 grayscale images of handwritten digits, each measuring 28 x 28 pixels and annotated by humans. 

# Body 
The body of this project is structured into four files.


1. `datasets.py`
2. `model.py`
3. `util.py`
4. `S6.ipynb` 

## Datasets

This code defines a function load_data(batch_size) that loads and preprocesses the MNIST dataset for training and testing a machine-learning model, specifically designed for use with PyTorch. The MNIST dataset consists of 28x28 pixel images of handwritten digits (0 through 9) and is widely used for benchmarking image processing systems. The function takes a single parameter, batch_size, which specifies the number of images to be processed in each batch. Here's a breakdown of what each part of the function does:

Train data transformations (train_transforms): This is a series of data preprocessing steps applied to the training dataset.

transforms.RandomApply([...], p=0.1): Randomly applies a list of transformations with a probability of p=0.1. Here, it's applying a centre crop to a 22x22 pixel region of the image.
transforms.Resize((28, 28)): Resize the cropped image back to 28x28 pixels.
transforms.RandomRotation((-15., 15.), fill=0): Randomly rotates the image by an angle between -15 and 15 degrees, filling the background with zeros (black).
transforms.ToTensor(): Converts the images to PyTorch tensors.
transforms.Normalize((0.1307,), (0.3081,)): Normalize the tensor images using the mean (0.1307) and standard deviation (0.3081) of the MNIST dataset. This is done for the single (grayscale) channel.
Test data transformations (test_transforms): This is similar to the train data transformations but simplified, as it's typically best practice not to augment the test dataset.

transforms.ToTensor(): Converts the images to PyTorch tensors.
transforms.Normalize((0.1307,), (0.3081,)): Normalize the tensor images using the same mean and standard deviation as the training dataset.
Loading the MNIST dataset:

The training and test datasets are loaded using datasets.MNIST with appropriate flags. The download=True flag ensures the dataset is downloaded if it's not already present in the specified directory ('../data'). The transform parameter applies the defined transformations to the datasets.
DataLoader:

In summary, this function sets up the data pipeline for training and testing on the MNIST dataset, implementing common preprocessing steps and data augmentation for the training set to improve model generalization.

## Model
Model Architecture 




## Util


## S6 Jupyter Notebook
