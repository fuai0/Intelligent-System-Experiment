import torch
from torch import nn
from torchvision import datasets, transforms

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="data/",train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),
                                                                                          transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size=64,shuffle=True
)
net = torch.load('data/model_MNIST.pth').cuda()

def test():
    net.eval()
    test_loss = 0
    corret = 0
    for data,target in test_loader:
        data = torch.tensor(data).type(torch.FloatTensor).cuda()
        target = torch.tensor(target).type(torch.LongTensor).cuda()
        output = net(data)
        test_loss = test_loss + nn.CrossEntropyLoss()(output,target).item()
        _,prod = output.max(1)
        corret = corret + prod.eq(target).sum().item()

    print("Test set:Average loss:{:.4f},Accuracy:{}/{}({:.0f}%)\n".format(test_loss/len(test_loader.dataset),corret,len(test_loader.dataset),100.*corret/len(test_loader.dataset)))


# import torch
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from modul import Net
#
# model = Net().cuda()
#
#
# test_batch_size = 1000
#
#
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST(root='data',train=False,
#                    transform=transforms.Compose([
#                    transforms.ToTensor(),
#                    transforms.Normalize((0.1307,),(0.3081,))])),
#     batch_size=test_batch_size,shuffle=True)
#
# def test():
#     model.load_state_dict(torch.load('data/model_MNIST.pth'))
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total_samples = 0
#     with torch.no_grad():
#         for data,target in test_loader:
#             data= data.type(torch.FloatTensor).cuda()
#             target = target.type(torch.LongTensor).cuda()
#             output = model(data)
#             loss = torch.nn.functional.cross_entropy(output, target)
#             test_loss += loss.item() * data.size(0)
#             _, predicted = output.max(1)
#             correct += predicted.eq(target).sum().item()
#             total_samples += data.size(0)
#     test_loss /= total_samples
#     accuracy = 100.0 * correct / total_samples
#     print('Test set:Average loss:{:.4f},Accuracy:{}/{}({:.0f}%)\n'.format(
#         test_loss,correct,total_samples,accuracy))

if __name__ == '__main__':
    test()