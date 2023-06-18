import numpy as np
import torch
from torch import nn

net = torch.load("net.pkl").cuda()
x = np.load("test_set.npy")
x = torch.tensor(x,dtype=torch.float32).cuda()
y = np.zeros(200)
for i in range(200):
    if i%2 ==0:
        y[i] = 0
    else:
        y[i] = 1
prod = []
for i in net(x):
    prod.append(torch.argmax(i).item())

print((prod==y).sum()/200)



# num = 0
# for i in range(100):
#     if y_hat[i] > 0.5:
#         y_hat[i] = 1
#     else:
#         y_hat[i] = 0
#     if y_hat[i] == y[i]:
#         num += 1
#
# print(f"accuraty:{num}%")

