#coding:utf8
from torch import nn
from .BasicModule import BasicModule


class MnistNet(BasicModule):
    def __init__(self, num_classes=10):
        super(MnistNet, self).__init__()
        self.model_name = 'mnistNet'
        self.features = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(28, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classfier = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x
