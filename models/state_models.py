import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision.models.resnet import BasicBlock, ResNet

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class Base_CNN(nn.Module):
    """
    Implementation of a standard CNN for experiments
    """

    def __init__(self, act = F.relu, out_act = F.softmax, no_weights = True):
        super(Base_CNN, self).__init__()

        self.act = act
        self.pooling = nn.MaxPool2d((2,2))
        self.out_act = out_act
        self.classification = True

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.fullyConnected = nn.Linear(in_features= 1600, out_features = 10)

        self.num_params = sum(p.numel() for p in self.parameters())
        self.param_shapes = [list(i.shape) for i in self.parameters()]

        self.buffer_shapes = [list(i.shape) for i in self.buffers()]

        self.apply(_weights_init)

        self._weights = None
        if no_weights:
            return
        
        self._weights = nn.ParameterList()
        for p in self.parameters():
            self._weights.append(p)

    def set_state_dict(self, params, buffers):

        param_keys = dict(self.named_parameters()).keys()
        buffer_keys = dict(self.named_buffers()).keys()

        sd = self.state_dict()
        param_dict = {}

        param_id = 0
        buffer_id = 0

        for key in sd.keys():
            if key in param_keys:
                param_dict[key] = params[param_id]
                param_id += 1
            elif key in buffer_keys:
                param_dict[key] = buffers[buffer_id]
                buffer_id += 1

        return param_dict
    
    def forward(self, x, params, buffers = None):

        custom_state_dict = self.set_state_dict(params, buffers)

        hidden = x

        hidden = F.conv2d(hidden, custom_state_dict["Conv1.weight"], bias = custom_state_dict["Conv1.bias"])
        hidden = self.act(hidden)
        hidden = self.pooling(hidden)
        
        hidden = F.conv2d(hidden, custom_state_dict["Conv2.weight"], bias = custom_state_dict["Conv2.bias"])
        hidden = self.act(hidden)
        hidden = self.pooling(hidden)
        
        hidden = torch.flatten(hidden, 1)
        hidden = F.linear(hidden, custom_state_dict["fullyConnected.weight"], bias = custom_state_dict["fullyConnected.bias"])

        return self.out_act(hidden), hidden



class ResNet18_CIFAR(ResNet):
    def __init__(self, num_classes=10, act = F.relu, out_act = F.softmax, no_weights = True):
        #ResNet18 uses BasicBlock and [2, 2, 2, 2] layers
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()

        self.act = act
        self.out_act = out_act
        self.classification = True

        self.num_params = sum(p.numel() for p in self.parameters())
        self.param_shapes = [list(i.shape) for i in self.parameters()]

        self.buffer_shapes = [list(i.shape) for i in self.buffers()]

        self.apply(_weights_init)

        self._weights = None
        if no_weights:
            return
        
        self._weights = nn.ParameterList()
        for p in self.parameters():
            self._weights.append(p)

    def set_state_dict(self, params, buffers):

        param_keys = dict(self.named_parameters()).keys()
        buffer_keys = dict(self.named_buffers()).keys()

        sd = self.state_dict()
        param_dict = {}

        param_id = 0
        buffer_id = 0

        for key in sd.keys():
            if key in param_keys:
                param_dict[key] = params[param_id]
                param_id += 1
            elif key in buffer_keys:
                param_dict[key] = buffers[buffer_id]
                buffer_id += 1

        return param_dict
    
    def manual_bn(self, custom_state_dict, keys, hidden, current_id):
        if self.training:
            custom_state_dict[keys[current_id+2]] = 0.9*custom_state_dict[keys[current_id+2]]+0.1*hidden.mean([0,2,3])
            custom_state_dict[keys[current_id+3]] = 0.9*custom_state_dict[keys[current_id+3]]+0.1*hidden.var([0, 2, 3], unbiased=False)
            custom_state_dict[keys[current_id+4]] += 1
        
        hidden = F.batch_norm(hidden, 
                                    custom_state_dict[keys[current_id+2]].detach(),
                                    custom_state_dict[keys[current_id+3]].detach(),
                                    custom_state_dict[keys[current_id]],
                                    custom_state_dict[keys[current_id+1]],
                                    training = False)

        return hidden
    
    def forward(self, x, params, buffers):

        modules = dict(self.named_modules())
        custom_state_dict = self.set_state_dict(params, buffers)
        keys = list(custom_state_dict.keys())
        key_id = 0

        hidden = x
        for_shortcut = None

        while key_id < len(keys):
            current_key = keys[key_id]
            current_key_split = current_key.split(".")

            if "conv1" in current_key_split or "conv2" in current_key_split:
                if "conv1" in current_key_split:
                    for_shortcut = hidden.clone()
                hidden = F.conv2d(hidden, custom_state_dict[current_key], stride = modules[".".join(current_key_split[:-1])].stride, padding = 1)
                key_id += 1
            elif "bn1" in current_key_split:
                hidden = self.manual_bn(custom_state_dict, keys, hidden, key_id)
                hidden = F.relu(hidden)
                key_id += 5
            elif "bn2" in current_key_split:
                hidden = self.manual_bn(custom_state_dict, keys, hidden, key_id)
                if "downsample" not in keys[key_id+5].split("."):
                    hidden = F.relu(hidden)
                key_id += 5
            elif "downsample" in current_key_split:
                shortcut = F.conv2d(for_shortcut, custom_state_dict[current_key], stride = modules[".".join(current_key_split[:-1])].stride)
                shortcut = self.manual_bn(custom_state_dict, keys, shortcut, key_id+1)
                hidden = F.relu(hidden+shortcut)
                key_id += 6
            elif "fc" in current_key_split:
                hidden = F.avg_pool2d(hidden, hidden.size()[3])
                hidden = hidden.view(hidden.size(0),-1)
                hidden = F.linear(hidden, custom_state_dict[current_key], bias = custom_state_dict[keys[key_id+1]])
                key_id += 2

        buffer_names = [name for name, _ in self.named_buffers()]
        for i in range(len(buffer_names)):
            buffers[i] = custom_state_dict[buffer_names[i]]
            
        return self.out_act(hidden), hidden