import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision.models.resnet import BasicBlock, ResNet

from models.BayesByBackpropLinearLayer import BayesByBackpropLayer
from models.BayesByBackprop2dConvLayer import BayesByBackprop2DConvLayer

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    
class BayesByBackprop_Base_CNN(nn.Module):
    
    def __init__(self,prior,posterior,device):
        super(BayesByBackprop_Base_CNN, self).__init__()
        self.activation = F.relu
        
        self.Conv1 = BayesByBackprop2DConvLayer(in_channels=1, out_channels=64, kernel_size=3,prior=prior,posterior=posterior, device = device)
        self.Conv2 = BayesByBackprop2DConvLayer(in_channels=64, out_channels=64, kernel_size=3,prior=prior,posterior=posterior, device = device)
        self.BBBLayer = BayesByBackpropLayer(indim=1600, outdim=10,prior=prior,posterior=posterior, device = device)
        self.pooling = nn.MaxPool2d((2,2))

        self.num_params = sum(p.numel() for p in self.parameters())

        self.apply(_weights_init)

    def all_layer_KL(self):
        log_prior = 0
        log_posterior = 0
        for module in self.modules():
            if hasattr(module, 'KL_divergence'):
                current_log_posterior, current_log_prior =  module.KL_divergence()
                log_posterior += current_log_posterior
                log_prior += current_log_prior
        return log_posterior, log_prior
    
    def forward(self, x):
        hidden = x

        hidden = self.Conv1(hidden)
        hidden = self.activation(hidden)
        hidden = self.pooling(hidden)
        
        hidden = self.Conv2(hidden)
        hidden = self.activation(hidden)
        hidden = self.pooling(hidden)
        
        hidden = torch.flatten(hidden, 1)
        hidden = self.BBBLayer(hidden)

        return hidden
    


class BayesByBackprop_ResNet18_CIFAR(ResNet):

    def __init__(self, prior, posterior,device):
        super().__init__(block = BasicBlock, layers = [2, 2, 2, 2], num_classes = 10)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()

        replace_with_bayesian_layers(self, prior, posterior, device)

        self.num_params = sum(p.numel() for p in self.parameters())

        self.apply(_weights_init)

    def all_layer_KL(self):
        log_prior = 0
        log_posterior = 0
        for module in self.modules():
            if hasattr(module, 'KL_divergence'):
                current_log_posterior, current_log_prior =  module.KL_divergence()
                log_posterior += current_log_posterior
                log_prior += current_log_prior
        return log_posterior, log_prior
    
    def forward(self, x):
        hidden = x
        hidden = super().forward(hidden)

        return hidden


def replace_with_bayesian_layers(model, prior, posterior, device):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):

            new_layer = BayesByBackprop2DConvLayer(
                module.in_channels, module.out_channels, module.kernel_size[0],
                prior = prior, posterior = posterior, stride = module.stride, padding = module.padding, device = device
            )
            setattr(model, name, new_layer)
        elif isinstance(module, nn.Linear):
            new_layer = BayesByBackpropLayer(module.in_features, module.out_features, prior = prior, posterior = posterior, device = device)
            setattr(model, name, new_layer)
        else:
            replace_with_bayesian_layers(module, prior, posterior, device)