import torch
import torch.nn as nn

# 训练集构建和转换
x = [[0,0],[0,1],[1,0],[1,1]]
y = [[0],[1],[1],[0]]
x_train = torch.tensor(x)
y_train = torch.tensor(y)
x_train = x_train.float().cuda()
y_train = y_train.float().cuda()

# 网络搭建
net = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,20),
    nn.ReLU(),
    nn.Linear(20,100),
    nn.ReLU(),
    nn.Linear(100,1),
    nn.Sigmoid()
).cuda()

# 优化器和损失函数
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
loss_func = nn.MSELoss()

# 训练部分
for epoch in range(5000):
    out = net(x_train)
    loss = loss_func(out,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%1000 == 0:
        print("迭代次数：{}".format(epoch))
        print("误差：{}".format(loss))

# 输出最后一次训练结果
out = net(x_train).cpu()
print("out:{}".format(out.data))

torch.save(net, 'net.pkl')



