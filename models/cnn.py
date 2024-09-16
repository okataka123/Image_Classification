import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    '''
    MNIST用（グレースケール画像）
    '''
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        #Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        #Linear(in_features, out_features)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x) # val of x.shape = torch.Size([100, 1, 28, 28])
        x = self.relu(x)  # val of x.shape = torch.Size([100, 16, 26, 26])
        x = self.pool(x)  # val of x.shape = torch.Size([100, 16, 26, 26])
        x = self.conv2(x) # val of x.shape = torch.Size([100, 16, 13, 13])
        x = self.relu(x)  # val of x.shape = torch.Size([100, 32, 11, 11])
        x = self.pool(x)  # val of x.shape = torch.Size([100, 32, 11, 11])
        #縦に並べる
        x = x.view(x.size()[0], -1) # val of x.shape = torch.Size([100, 32, 5, 5])
        x = self.fc1(x)   # val of x.shape = torch.Size([100, 800])
        x = self.relu(x)  # val of x.shape = torch.Size([100, 120])
        x = self.fc2(x)   # val of x.shape = torch.Size([100, 120])
        # output of x.shape = torch.Size([100, 10])
        return x
