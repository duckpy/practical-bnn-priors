import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision.models.resnet import BasicBlock, ResNet

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class Base_CNN(nn.Module):
    
    def __init__(self):
        super(Base_CNN, self).__init__()
        self.activation = F.relu
        
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc = nn.Linear(in_features= 1600, out_features = 10)
        self.pooling = nn.MaxPool2d((2,2))

        self.apply(_weights_init)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.activation(x)
        x = self.pooling(x)
        
        x = self.Conv2(x)
        x = self.activation(x)
        x = self.pooling(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    


class ResNet18_CIFAR(ResNet):
    def __init__(self, num_classes = 10):
        super().__init__(block = BasicBlock, layers = [2,2,2,2], num_classes = num_classes)

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.maxpool = nn.Identity()

        self.apply(_weights_init)

    def forward(self, x):
        x = super().forward(x)
        return x