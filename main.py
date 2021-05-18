import torch
from torchvision import transforms, datasets
from Scripts.pytorch.lenet5_pytorch import *


def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize
    ])

    trainset = datasets.CIFAR10(root='/ data /', train=True, download=True, transform=transform)  # training set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    testset = datasets.CIFAR10('/ data /', train=False, download=True, transform=transform)  # test set

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def train_model():
    trainloader, testloader, _ = get_data()
    lenet = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lenet.parameters(), lr=0.001)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = lenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    PATH = './cifar_net.pth'
    torch.save(lenet.state_dict(), PATH)
    print('Finished Training')


train_model()
