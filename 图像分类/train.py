import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from modul import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_set = datasets.CIFAR10(
    root="data",train=True,download=False,transform=transform
)
train_loader = DataLoader(train_set,batch_size=64,shuffle=True)

net = Net().cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters())

def train(epoch):
    for batch_index,(data,target) in enumerate(train_loader):
        data = torch.tensor(data).type(torch.FloatTensor).cuda()
        target = torch.tensor(target).type(torch.LongTensor).cuda()
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out,target)
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 0:
            print("训练轮数：{}[{}/{}({:.0f}%)]\tLoss:{:.6f}".format(epoch, batch_index * len(data),
                    len(train_loader.dataset),100. * batch_index / len(train_loader),loss.item()))
    if epoch == 9:
        torch.save(net,"net.pth")

if __name__ == "__main__":
    for i in range(10):
        train(i)

