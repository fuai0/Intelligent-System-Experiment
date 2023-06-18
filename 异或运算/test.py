import torch

# 测试集构建和转换
x = [[1,1]]
x_test = torch.tensor(x)
x_test = x_test.float().cuda()

# 导入训练好的网络并使用
net = torch.load('net.pkl')
outfinal = net(x_test)

# 对输出结果进行转换
if outfinal >=0.5:
    outfinal = 1
else:
    outfinal = 0

print("outfinal:{}".format(outfinal))