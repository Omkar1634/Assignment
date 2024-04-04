import torch.nn.functional as F
from torchsummary import summary
import torch.nn as  nn


def model_summary(model, input_size):
    """
    This function displays a summary of the model, providing information about its architecture,
    layer configuration, and the number of parameters it contains.
    :param model: model
    :param input_size: input_size for model
    :return:
    """
    summary(model, input_size=input_size)


class S9_model(nn.Module):
    def __init__(self):
        super(S9_model,self).__init__()


        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=128,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=32),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=2,dilation=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=32),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=2,dilation=4),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=32),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=2,dilation=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,groups=16),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=2,dilation=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=32,out_features=10)


    def forward(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = F.log_softmax(x,dim=1)
        return x