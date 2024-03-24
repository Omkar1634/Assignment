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


class S8_Model_GN(nn.Module):
    def __init__(self,group_number=None):
        super(S8_Model_GN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=7, padding=3, bias=False), 
            nn.GroupNorm(group_number,10),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1,bias=False),  
            nn.GroupNorm(group_number,20),
            nn.ReLU(),
            nn.Dropout(0.10)
            
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1),  
            
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=12, kernel_size=3, padding=1, bias=False),  
            nn.GroupNorm(group_number,12),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1,bias=False),  
            nn.GroupNorm(group_number,12),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1,bias=False),  
            nn.GroupNorm(group_number,12),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv1x1_7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=1), 
           
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,bias=False),  
            nn.GroupNorm(group_number,32),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,bias=False),  
            nn.GroupNorm(group_number,32),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,bias=False),  
            nn.GroupNorm(group_number,64),
            nn.Dropout(0.10),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1))


        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1x1_3(x)
        x = self.pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv1x1_7(x)
        x = self.pool_2(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = self.conv1x1_11(x)
        x = x.view(x.size(0), -1)  
        x = F.log_softmax(x, dim=1)
        return x


class S8_Model_BN(nn.Module):
    def __init__(self):
        super(S8_Model_BN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, padding=1,bias=False),  
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.10)
            
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1),  
            
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=12, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1,bias=False),  
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1,bias=False),  
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv1x1_7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=1), 
           
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,bias=False),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,bias=False),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.10)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,bias=False),  
            nn.BatchNorm2d(64),
            nn.Dropout(0.10),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1))


        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1x1_3(x)
        x = self.pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv1x1_7(x)
        x = self.pool_2(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = self.conv1x1_11(x)
        x = x.view(x.size(0), -1)  
        x = F.log_softmax(x, dim=1)
        return x



class S8_Model_LN(nn.Module):
    def __init__(self):
        super(S8_Model_LN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, padding=3, bias=False),
            nn.LayerNorm([8,32,32 ]), 
            nn.ReLU(),
            nn.Dropout(0.10),

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.LayerNorm([4, 32,32]), 
            nn.ReLU(),
             nn.Dropout(0.10),

        )

        self.conv1x1_3 = nn.Sequential(
                        nn.Conv2d(4, 8, kernel_size=1),

        )

        self.pool_1 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.LayerNorm([4,16,16]),  # Adjust for post-pooling size
            nn.ReLU(),
                        nn.Dropout(0.10),

        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(4, 10, kernel_size=3, padding=1),
            nn.LayerNorm([10,16,16]),
            nn.ReLU(),
                        nn.Dropout(0.10),

        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(10, 8, kernel_size=3, padding=1),
            nn.LayerNorm([8, 16,16]),
            nn.ReLU(),
                        nn.Dropout(0.10),

        )

        self.conv1x1_7 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1),
            
        )

        self.pool_2 = nn.MaxPool2d(2, 2)
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(8, 10, kernel_size=3, padding=1),
            nn.LayerNorm([10,8,8]),
                        nn.Dropout(0.10),
                        nn.ReLU(),

        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.LayerNorm([16,8,8]),
                        nn.Dropout(0.10),
                        nn.ReLU(),

        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=1, padding=1),
                        nn.Dropout(0.10),
                        nn.LayerNorm([4,10,10]),
                        nn.ReLU(),

            
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_11 = nn.Sequential(
            nn.Conv2d(4, 10, kernel_size=1),
            
        )

    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1x1_3(x)
        x = self.pool_1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv1x1_7(x)
        x = self.pool_2(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = self.conv1x1_11(x)
        x = x.view(x.size(0), -1)  
        x = F.log_softmax(x, dim=1)
        return x
