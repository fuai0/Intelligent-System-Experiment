from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,8,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8,8,kernel_size=3,padding=1)
        self.max = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(64*64*8,1000)
        self.lin2 = nn.Linear(1000,1000)
        self.lin3 = nn.Linear(1000,2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.flat(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x


