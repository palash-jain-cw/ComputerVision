import torch.nn as nn

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


