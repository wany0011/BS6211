import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class AvgBN(nn.Module):
    def __init__(self, channel):
        super(AvgBN, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=1)
        self.channel = channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.channel, self.channel*2, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(self.channel*2, self.channel*2, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(self.channel*2, self.channel*4, kernel_size=(3, 3), padding=(1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.channel*4 + 1, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x, t):
        # print(x.shape, t.shape)
        x = self.bn1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.gap(x)

        # print(x.shape, t.shape)
        x = x.view(-1, self.channel*4)
        # print(x.shape, t.shape)
        x = torch.cat((x, t), 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = torch.cat((x, t), 1)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_features=1)
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.fc1 = nn.Linear(in_features=512+1, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)
