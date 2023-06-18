import torch
from torch import nn

net = torch.load("net.pkl")

x_test = [[1,0]]
x_test = torch.tensor(x_test,dtype=torch.float32).to("cuda:0")

y_test = [[1]]
y_test = torch.tensor(y_test,dtype=torch.float32).to("cuda:0")

y_pred = net(x_test)
print(y_pred)