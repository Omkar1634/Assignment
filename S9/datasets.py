import torch
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)['image']

def load_data_S9():
    train_transform = AlbumentationsTransform(A.Compose([
        A.HorizontalFlip(p=1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.CoarseDropout(max_holes=1, min_holes=1, max_height=16, min_height=16, max_width=16, min_width=16, fill_value=0, mask_fill_value=None),
        A.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
        ToTensorV2(),
    ]))

    test_transform = AlbumentationsTransform(A.Compose([
        A.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
        ToTensorV2(),
    ]))

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    kwargs = {'batch_size': 512, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if cuda else {'shuffle': True, 'batch_size': 64}

    train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10('../data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return train_loader, test_loader





# import torch
# from torchvision import transforms,datasets
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# def load_data_S9():

#     train_transform = A.Compose([
#         A.HorizontalFlip(always_apply=True,p=0.75),
#         A.ShiftScaleRotate(),
#         A.CoarseDropout(max_holes=1,min_holes=1,max_height=16,min_height=16,max_width=16,min_width=16,fill_value=0,mask_fill_value=None),
#         ToTensorV2(),
#         A.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))
#     ])


#     test_transform = A.Compose([
#         ToTensorV2(),
#         A.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))
#     ])

#     SEED = 1

#     # CUDA?
#     cuda = torch.cuda.is_available()
#     print("CUDA Available?", cuda)

#     # For reproducibility
#     torch.manual_seed(SEED)

#     if cuda:
#         torch.cuda.manual_seed(SEED)

#     kwargs = {'batch_size': 512, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if cuda else dict(shuffle=True, batch_size=64)
    
#     train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transform)
#     test_data = datasets.CIFAR10('../data', train=False, download=True, transform=test_transform)

#     test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

#     train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

#     return train_loader,test_loader
