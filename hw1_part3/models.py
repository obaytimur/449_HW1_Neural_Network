import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os.path
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch.optim as optim


class mlp_1(nn.Module):
    def __init__(self):
        super(mlp_1, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class mlp_2(nn.Module):
    def __init__(self):
        super(mlp_2, self).__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class cnn_3(nn.Module):
    def __init__(self):
        super(cnn_3, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 3 * 3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class cnn_4(nn.Module):
    def __init__(self):
        super(cnn_4, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 3 * 3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class cnn_5(nn.Module):
    def __init__(self):
        super(cnn_5, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3))
        self.fc1 = nn.Linear(16 * 3 * 3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class mlp_1_sig(nn.Module):
    def __init__(self):
        super(mlp_1_sig, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.logsigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class mlp_2_sig(nn.Module):
    def __init__(self):
        super(mlp_2_sig, self).__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.logsigmoid(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class cnn_3_sig(nn.Module):
    def __init__(self):
        super(cnn_3_sig, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 3 * 3, 10)

    def forward(self, x):
        x = F.logsigmoid(self.conv1(x))
        x = self.pool(F.logsigmoid(self.conv2(x)))
        x = self.pool(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class cnn_4_sig(nn.Module):
    def __init__(self):
        super(cnn_4_sig, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 3 * 3, 10)

    def forward(self, x):
        x = F.logsigmoid(self.conv1(x))
        x = F.logsigmoid(self.conv2(x))
        x = self.pool(F.logsigmoid(self.conv3(x)))
        x = self.pool(F.logsigmoid(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class cnn_5_sig(nn.Module):
    def __init__(self):
        super(cnn_5_sig, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3))
        self.fc1 = nn.Linear(16 * 3 * 3, 10)

    def forward(self, x):
        x = F.logsigmoid(self.conv1(x))
        x = F.logsigmoid(self.conv2(x))
        x = F.logsigmoid(self.conv3(x))
        x = self.pool(F.logsigmoid(self.conv4(x)))
        x = F.logsigmoid(self.conv5(x))
        x = self.pool(F.logsigmoid(self.conv6(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
