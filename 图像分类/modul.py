import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self,in_channels):
        super(Inception,self).__init__()
        self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = nn.Conv2d(24,24,kernel_size=3,padding=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5,padding=2)

        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self,x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch1x1 = self.branch1x1(x)

        branch_pool = nn.functional.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch3x3,branch5x5,branch1x1,branch_pool]
        return torch.cat(outputs,dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,10,kernel_size=5)
        self.conv2 = nn.Conv2d(88,20,kernel_size=5)

        self.incep1 = Inception(10)
        self.incep2 = Inception(20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(2200,10)

    def forward(self,x):
        x = nn.functional.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = nn.functional.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x