from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self,input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.lin(x)
        return x

# import torch.nn as nn
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1,6,5,1,2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(6,16,5),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#         )
#         self.den = nn.Sequential(
#             nn.Linear(16 * 5 * 5, 120),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84,10)
#         )
#
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size()[0],-1)
#         x = self.den(x)
#         return x