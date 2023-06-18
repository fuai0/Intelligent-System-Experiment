# 导入模板
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 构造样本集
data = torch.ones(100,2)
x0 = torch.normal(2*data,1)
x1 = torch.normal(-2*data,1)

# 数据合并
x = torch.cat((x0,x1),0).type(torch.FloatTensor)

# 标签
y0 = torch.zeros(100)
y1 = torch.zeros(100)
y = torch.cat((y0,y1),0).type(torch.LongTensor)

# 构建网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(2,15),
            nn.ReLU(),
            nn.Linear(15,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        classification = self.classify(x)
        return classification

# 训练网络
net = Net()
optimizer = torch.optim.SGD(net.parameters(),lr=0.03)
loss_func = nn.CrossEntropyLoss()

plt.ion()
for epoch in range(100):
    out = net(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%2 == 0:
        plt.cla()
        # 可视化
        classification = torch.max(out,1)[1]
        class_y = classification.data.numpy()

        target_y = y.data.numpy()

        # 绘制散点图
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=class_y,s=100,cmap="RdYlGn")

        accuracy = sum(class_y == target_y)/200
        plt.text(1.5,-4,"Accuracy={}".format(accuracy),fontdict={"size":20,"color":"red"})
        plt.pause(1)
        if accuracy == 1:
            break



plt.ioff()
plt.show()