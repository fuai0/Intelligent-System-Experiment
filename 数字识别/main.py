from torch import nn,optim
import numpy as np
import torch
from torchvision import datasets,transforms
from torch.utils.data import Dataset
from modul import Net

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="data/",train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),
                                                                                         transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size=64,shuffle=True)

epochs = 10
net = Net().cuda()

def train(epochs):
    net.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data = torch.tensor(data).type(torch.FloatTensor).cuda()
        target = torch.tensor(target).type(torch.LongTensor).cuda()
        optim.SGD(net.parameters(), lr=0.01, momentum=0.5).zero_grad()
        output = net(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optim.SGD(net.parameters(), lr=0.01, momentum=0.5).step()
        if batch_index % 10 == 0:
            print("训练轮数：{}[{}/{}({:.0f}%)]\tLoss:{:.6f}".format(epochs, batch_index * len(data),
                                                                    len(train_loader.dataset),
                                                                    100. * batch_index / len(train_loader),
                                                                    loss.item()))

    torch.save(net,'data/model_MNIST.pth')

if __name__ == "__main__":
    for i in range(epochs):
        train(i)


# import torch
# import torch.nn as nn
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from modul import Net
#
#
# model = Net().cuda()
#
#
# epochs = 10
# batch_size = 64
# lr = 0.01
# momentum = 0.5
# log_interval = 10
#
#
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST(root='data',train=True,download=True,
#                    transform=transforms.Compose([
#                    transforms.ToTensor(),
#                    transforms.Normalize((0.1307,),(0.3081,))])),
#     batch_size=batch_size,shuffle=True)
#
# def train(epoch):
#     model.train()
#     for batch_idx,(data,target)in enumerate(train_loader):
#         data = data.type(torch.FloatTensor).cuda()
#         target = target.type(torch.LongTensor).cuda()
#         torch.optim.SGD(model.parameters(),lr=lr,
#         momentum=momentum).zero_grad()
#         output = model(data)
#         loss = nn.CrossEntropyLoss()(output,target)
#         loss.backward()
#         torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum).step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
#                 epoch,batch_idx*len(data),len(train_loader.dataset),
#                 100.*batch_idx/len(train_loader),loss.item()))
#     torch.save(model, 'data/model_MNIST.pth')
#
# if __name__ == '__main__':
#     for epoch in range(epochs):
#         train(epoch)
