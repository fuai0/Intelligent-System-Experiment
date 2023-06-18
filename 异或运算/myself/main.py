import torch
from torch import nn,optim


x = [[0,0],[1,0],[0,1],[1,1]]
y = [[0],[1],[1],[0]]
x_train = torch.tensor(x,dtype=torch.float32).to("cuda:0")
y_train = torch.tensor(y,dtype=torch.float32).to("cuda:0")

net = nn.Sequential(
    nn.Linear(2,20),
    nn.ReLU(),
    nn.Linear(20,1),
    nn.Sigmoid()
).to("cuda:0")

loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),lr=0.1)

for i in range(1000):
    optimizer.zero_grad()
    y_pred = net(x_train)
    loss = loss_fn(y_pred,y_train)
    loss.backward()
    optimizer.step()

torch.save(net,"net.pkl")