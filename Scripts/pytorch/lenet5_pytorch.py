import torch
import torch.nn as nn
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report


# LeNet architecture (for Cifar10 dataset)
# 3x32x32 Input Image -> (5x5), s=1, p=0  -> avg pool s=2, p = 0 -> (5x5),s=1,p=0 -> avg pool s=2, p = 0
# -> Conv 5x5 to 120 channels x Linear 120 -> 84 x Linear 10
# Original LeNet uses sigmoid and tanh activation functions, I use ReLu here for better performance
# Originally, input images for LeNet were 1x32x32 (non-colored images), since we are using CIFAR10 dataset
# which contains 3 channel color images we are using input images dimensions as 3*32*32


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))  # num_examples * 120 * 1 * 1 -> num_examples * 120
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def get_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize
    ])

    trainset = datasets.CIFAR10(root='../../data/', train=True, download=True, transform=transform)  # training set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10('../../data/', train=False, download=True, transform=transform)  # test set

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def train_model(trainloader, learning_rate, batch_size, num_epochs):
    # Initialize the model
    lenet = LeNet()
    print('Model Architecture Initialized Successfully')

    # Define Loss
    criterion = nn.CrossEntropyLoss()

    # Define Optimizer
    optimizer = torch.optim.Adam(lenet.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # Send input and labels to GPU
            inputs.to(device)
            labels.to(device)

            # Forward Propagation
            outputs = lenet(inputs)
            loss = criterion(outputs, labels)

            # Backward Propagation
            optimizer.zero_grad()
            loss.backward()

            # Optimizer Step
            optimizer.step()

            # Loss
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    # PATH = '../models/cifar_lenet.pth'
    # torch.save(lenet.state_dict(), PATH)
    print('Finished Training')
    return lenet


def predict(model, loader):
    predictions, targets = [], []
    for images, labels in testloader:
        logps = model(images)
        output = torch.exp(logps)
        pred = torch.argmax(output, 1)
        # convert to numpy arrays
        pred = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        for i in range(len(pred)):
            predictions.append(pred[i])
            targets.append(labels[i])
    return predictions, targets


def compute_metrics(model, trainloader, testloader):
    train_pred, train_y = predict(model, trainloader)
    test_pred, test_y = predict(model, testloader)
    print('For Training Data')
    print(classification_report(train_y, train_pred))
    print('\n')
    print('For Test Set')
    print(classification_report(test_y, test_pred))


# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} for training'.format('cuda' if torch.cuda.is_available() else 'cpu'))

# Hyperparameters
learning_rate = 0.001
batch_size = 128
num_epochs = 2

trainloader, testloader, _ = get_data(batch_size=batch_size)
print('Data Loaded Successfully')

model = train_model(trainloader,learning_rate, batch_size, num_epochs)
compute_metrics(model, trainloader, testloader)
