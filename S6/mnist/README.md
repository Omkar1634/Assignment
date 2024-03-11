# Handwritten Digit Recognition using Convolutional Neural Networks with Pytorch

The MNIST dataset comprises 70,000 grayscale images of handwritten digits, each measuring 28 x 28 pixels and annotated by humans. 

# Body 
The body of this project is structured into four files.
1. `datasets.py`
2. `model.py`
3. `util.py`
4. `S6.ipynb` 

## Datasets.py

This code defines a function load_data(batch_size) that loads and preprocesses the MNIST dataset for training and testing a machine-learning model, specifically designed for use with PyTorch. The MNIST dataset consists of 28x28 pixel images of handwritten digits (0 through 9) and is widely used for benchmarking image processing systems. The function takes a single parameter, batch_size, which specifies the number of images to be processed in each batch. Here's a breakdown of what each part of the function does:

Below are the parameters that we have used to transform the image.

1. Transforms.RandomApply: Randomly applies a list of transformations with a probability.
2. Transforms.Resize: Resize the cropped image back to 28x28 pixels.
3. Transforms.RandomRotation: Randomly rotates the image by an angle degree, filling the background with zeros (black).
4. Transforms.ToTensor: Converts the images to PyTorch tensors.
5. Transforms. Normalize: Normalize the tensor images using the mean and standard deviation of the MNIST dataset. This is done for the single (grayscale) channel.

### Dataloader 
The training and test datasets are loaded using datasets.MNIST with appropriate flags. The download=True flag ensures the dataset is downloaded if it's not already present in the specified directory ('../data'). The transform parameter applies the defined transformations to the datasets.
DataLoader:


## Model.py
Model Architecture 
![1](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/345013c7-47c1-4fe5-a6e0-5d841b5a0eb0)

Model Summary
![2](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/ef183a99-1d92-4a17-a6a1-7268064744e8)


## Util.py
This code provides a comprehensive setup for training and evaluating a neural network model using PyTorch. It is structured to work with datasets that can be loaded in batches (for instance, using PyTorch's DataLoader), and it's particularly suited for tasks like image classification. The code is organized into functions that handle different aspects of the training and evaluation process:

get_correct_pred_count: This utility function calculates the number of correct predictions by comparing the predicted labels with the true labels.

train: This function orchestrates the training process for one epoch. It iterates over the training dataset, computes the loss for each batch using a specified loss criterion, and updates the model's weights using backpropagation. It also accumulates and reports the training loss and accuracy.

test: Similar to train, but for evaluating the model's performance on a test dataset. It disables gradient computations (to save memory and computations) and accumulates the test loss and accuracy. The model is set to evaluation mode (model.eval()) to ensure that operations like dropout are disabled during testing.

plot_acc_loss: This function visualizes the training process by plotting both the training and test loss and accuracy as a function of the training epoch. This is useful for monitoring the model's learning progress and diagnosing issues like overfitting or underfitting.

The script also initializes lists to keep track of the training and test losses and accuracies (train_losses, test_losses, train_acc, test_acc), which are updated during the training and testing phases and used for plotting the performance metrics.

Overall, this setup is a typical workflow in machine learning and deep learning for training models, evaluating their performance on a held-out test set, and visualizing their learning progress. It leverages PyTorch's functionalities for model training, including automatic differentiation for backpropagation and DataLoader for efficient batch processing of datasets.

## S6 Jupyter Notebook
![image](https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/ea514dc8-2b75-4393-8a4d-05e9d54b940f)

