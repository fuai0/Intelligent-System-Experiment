from torch import nn, optim

from net_model import Net
import numpy as np
import torch

samples = 2000
minibatch = 20
pic_size = 128

train_data = np.load("train_set.npy")
train_data = torch.tensor(train_data,dtype=torch.float32).cuda()

x = np.zeros(minibatch*3*pic_size*pic_size)
x = np.resize(x,(minibatch,3,pic_size,pic_size))
x = torch.tensor(x,dtype=torch.float32).cuda()
y = np.zeros(minibatch)
y = torch.tensor(y,dtype=torch.int64).cuda()
net = Net().cuda()


loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(),lr=0.001)


for epoch in range(100):
    for iterations in range(samples//minibatch):
        k = 0
        for i in range(iterations*minibatch,iterations*minibatch+minibatch):
            x[k,0,:,:] = train_data[i,0,:,:]
            x[k,1,:,:] = train_data[i,1,:,:]
            x[k,2,:,:] = train_data[i,2,:,:]
            if i%2 == 1:
                y[k] = 1
            else:
                y[k] = 0
            k = k+1

        out = net(x)
        print(out)
        loss = loss_fn(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        print(loss.item())
        print(f"第{epoch}轮训练进度:{(iterations+1)*100/(samples/minibatch)}%")

torch.save(net,"net.pkl")
