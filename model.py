import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class S6_Model_1(nn.Module):
    def __init__(self):
        super(S6_Model_1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=28, kernel_size=3, padding=1),  
            nn.BatchNorm2d(28),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1), 
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  
            nn.ReLU(),
            
            
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 10, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),  
        )
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1x1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv1x1_2(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)  
        x = F.log_softmax(x, dim=1)
        return x



class S7_Model(nn.Module):
    def __init__(self):
        super(S6_Model_1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=28, kernel_size=3, padding=1),  
            nn.BatchNorm2d(28),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1), 
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  
            nn.ReLU(),
            
            
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 10, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),  
        )
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1x1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv1x1_2(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)  
        x = F.log_softmax(x, dim=1)
        return x






def model_summary(model, input_size):
    """
    This function displays a summary of the model, providing information about its architecture,
    layer configuration, and the number of parameters it contains.
    :param model: model
    :param input_size: input_size for model
    :return:
    """
    summary(model, input_size=input_size)
