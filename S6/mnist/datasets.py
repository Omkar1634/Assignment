import torch.utils.data
from torchvision import datasets, transforms

def load_data(batch_size):


    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)), # Resize image at (28,28)
        transforms.RandomRotation((-15., 15.), fill=0), # Rotate the image by angle.
        transforms.ToTensor(),# Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,)), # Normalize images
        ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(), # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize images
        ])


    
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

    return train_loader,test_loader