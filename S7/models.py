import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def model_summary(model, input_size):
    """
    This function displays a summary of the model, providing information about its architecture,
    layer configuration, and the number of parameters it contains.
    :param model: model
    :param input_size: input_size for model
    :return:
    """
    summary(model, input_size=input_size)



class S7_Model_1(nn.Module):
    def __init__(self):
        super(S7_Model_1, self).__init__()

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


class S7_Model_2(nn.Module):
    def __init__(self):
        super(S7_Model_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25)

            )
        
        self.trans_block1 = nn.Sequential(
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 16, 1), 
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, padding=2),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.25)
            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),  
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)  
        )
        self.trans_block2 = nn.Sequential(
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(4, 32, 1), 
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),     
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1),
            nn.ReLU(),
        )
        

        
        
        # self.conv1x1_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, padding=1), 
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(),
        # )
    def forward(self, x):
            x = self.conv1(x)
            x = self.trans_block1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.trans_block2(x)
            x = self.conv4(x)
            x = self.conv5(x)

            x = x.view(x.size(0), -1)  
            x = F.log_softmax(x, dim=1)
            return x
    


class S7_Model_3(nn.Module):
    def __init__(self):
        super(S7_Model_3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3, bias=False), 
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5, padding=2),  
            nn.BatchNorm2d(10),
            nn.ReLU(),
            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),  
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),     
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 10, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),  
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=1)
        )

        self.dropout = nn.Dropout(0.25)
        
        # self.conv1x1_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, padding=1), 
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(),
        # )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv1x1(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)

        x = x.view(-1,10)  
        x = F.log_softmax(x, dim=1)
        return x


class S7_Model_4(nn.Module):
    def __init__(self):
        super(S7_Model_4, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, padding=0),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        # transition block 1
        self.trans_block1 = nn.Sequential(
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(8, 4, 1), 
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=0),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=0), 
            nn.ReLU(),  
            nn.BatchNorm2d(8),
            nn.Dropout(0.2)  
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=1),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=1),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(4,8, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # This reduces each 16-channel feature map to 1x1
        )
        self.fc = nn.Linear(8, 10)  # There are 16 channels coming from conv7, and 10 output classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans_block1(x)
        #x = self.conv3(x)
        #x = self.conv4(x)
        #x = self.conv5(x)
        #x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x



class S7_Model_5(nn.Module):
    def __init__(self):
        super(S7_Model_5, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0, bias=False), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, padding=0),  
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        # transition block 1
        self.trans_block1 = nn.Sequential(
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(10, 12, 1), 
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=0),  
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d( 12,4, kernel_size=3, padding=0), 
            nn.ReLU(),  
            nn.BatchNorm2d(4),
            #nn.Dropout(0.2)  
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(8, 4, kernel_size=1),
        #     nn.ReLU(),
        # )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(4, 8, kernel_size=1),
        #     nn.ReLU(),
        # )
        self.conv7 = nn.Sequential(
            nn.Conv2d(4,8, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # This reduces each 16-channel feature map to 1x1
        )
        self.fc = nn.Linear(8, 10)  # There are 16 channels coming from conv7, and 10 output classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans_block1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.conv5(x)
        #x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x



