import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

test_set = datasets.CIFAR10(
    root="data",train=False,download=True,transform=transform
)
test_loader = DataLoader(test_set,batch_size=64,shuffle=True)

net = torch.load("net.pth").cuda()

def test():
    test_loss = 0
    correct = 0
    for data,target in test_loader:
        data = torch.tensor(data).type(torch.FloatTensor).cuda()
        target = torch.tensor(target).type(torch.LongTensor).cuda()
        output = net(data)
        test_loss = test_loss + nn.CrossEntropyLoss()(output, target).item()
        _, prod = output.max(1)
        correct = correct + prod.eq(target).sum().item()
    print("Test set:Average loss:{:.4f},Accuracy:{}/{}({:.0f}%)\n".format(test_loss / len(test_loader.dataset),
                                                                                correct, len(test_loader.dataset),
                                                                                100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    test()