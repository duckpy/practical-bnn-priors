import os

import torch
import torch.nn.functional as F

from torchvision import datasets
from torchvision.transforms import ToTensor

def generate_dataset(dataset, device, is_one_hot = False):
    """
    Generate a dataset
    """
    if dataset == "mnist":
        data = MNISTData(os.getcwd() + "/data", device, is_one_hot = is_one_hot)
        return data
    elif dataset == "fashionmnist":
        data = FashionMNISTData(os.getcwd() + "/data", device, is_one_hot = is_one_hot)
        return data
    elif dataset == "CIFAR10":
        data = CIFAR10Data(os.getcwd() + "/data", device, is_one_hot = is_one_hot)
        return data
    
class FastTensorDataLoader:
    """
    Source: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    and https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset = tensors[0]

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class MNISTData():
    def __init__(self, data_path, device, validation_size = 0, is_one_hot = False):
        self.device = device
        self.num_train = 60000
        self.is_one_hot = is_one_hot
        self.validation_size = validation_size

        self.train_inputs, self.train_outputs = self.init_train(data_path)

        self.test_inputs, self.test_outputs, self.validation_inputs, self.validation_outputs = self.init_test_val(data_path)

    def init_train(self, data_path):
        self.training_data = datasets.MNIST(root=data_path,train=True,download=True,transform=ToTensor())
        train_inputs = torch.empty((60000,1,28,28))
        train_outputs = torch.empty((60000), dtype = torch.long)

        for idx, t in enumerate(self.training_data):
            data,label = t
            train_inputs[idx] = data
            train_outputs[idx] = label

        #One_hot
        if self.is_one_hot:
            train_outputs = F.one_hot(train_outputs, num_classes = 10).float()

        return train_inputs.to(self.device), train_outputs.to(self.device)
    
    def init_test_val(self, data_path):
        self.test_data = datasets.MNIST(root=data_path,train=False,download=True,transform=ToTensor())
        test_inputs = torch.empty((10000,1,28,28))
        test_outputs = torch.empty((10000), dtype = torch.long)

        for idx, t in enumerate(self.test_data):
            data,label = t
            test_inputs[idx] = data
            test_outputs[idx] = label

        #One_hot
        if self.is_one_hot:
            test_outputs = F.one_hot(test_outputs, num_classes = 10).float()
            return test_inputs[self.validation_size:,:,:,:].to(self.device), test_outputs[self.validation_size:,:].to(self.device), test_inputs[:self.validation_size,:,:,:].to(self.device), test_outputs[:self.validation_size,:].to(self.device)

        return test_inputs[self.validation_size:,:,:,:].to(self.device), test_outputs[self.validation_size:].to(self.device), test_inputs[:self.validation_size,:,:,:].to(self.device), test_outputs[:self.validation_size].to(self.device)

    def get_train_dataloader(self, batch_size = 32):
        return FastTensorDataLoader(self.train_inputs, self.train_outputs, batch_size = batch_size, shuffle = True)

    def get_test_dataloader(self, batch_size = 32):
        return FastTensorDataLoader(self.test_inputs, self.test_outputs, batch_size = batch_size, shuffle = True)

class FashionMNISTData():
    def __init__(self, data_path, device, validation_size = 0, is_one_hot = False):
        self.device = device
        self.num_train = 60000
        self.is_one_hot = is_one_hot
        self.validation_size = validation_size

        self.train_inputs, self.train_outputs = self.init_train(data_path)

        self.test_inputs, self.test_outputs, self.validation_inputs, self.validation_outputs = self.init_test_val(data_path)

    def init_train(self, data_path):
        self.training_data = datasets.FashionMNIST(root=data_path,train=True,download=True,transform=ToTensor())
        train_inputs = torch.empty((60000,1,28,28))
        train_outputs = torch.empty((60000), dtype = torch.long)

        for idx, t in enumerate(self.training_data):
            data,label = t
            train_inputs[idx] = data
            train_outputs[idx] = label

        #One_hot
        if self.is_one_hot:
            train_outputs = F.one_hot(train_outputs, num_classes = 10).float()

        return train_inputs.to(self.device), train_outputs.to(self.device)
    
    def init_test_val(self, data_path):
        self.test_data = datasets.FashionMNIST(root=data_path,train=False,download=True,transform=ToTensor())
        test_inputs = torch.empty((10000,1,28,28))
        test_outputs = torch.empty((10000), dtype = torch.long)

        for idx, t in enumerate(self.test_data):
            data,label = t
            test_inputs[idx] = data
            test_outputs[idx] = label

        #One_hot
        if self.is_one_hot:
            test_outputs = F.one_hot(test_outputs, num_classes = 10).float()
            return test_inputs[self.validation_size:,:,:,:].to(self.device), test_outputs[self.validation_size:,:].to(self.device), test_inputs[:self.validation_size,:,:,:].to(self.device), test_outputs[:self.validation_size,:].to(self.device)

        return test_inputs[self.validation_size:,:,:,:].to(self.device), test_outputs[self.validation_size:].to(self.device), test_inputs[:self.validation_size,:,:,:].to(self.device), test_outputs[:self.validation_size].to(self.device)

    def get_train_dataloader(self, batch_size = 32):
        return FastTensorDataLoader(self.train_inputs, self.train_outputs, batch_size = batch_size, shuffle = True)

    def get_test_dataloader(self, batch_size = 32):
        return FastTensorDataLoader(self.test_inputs, self.test_outputs, batch_size = batch_size, shuffle = True)

class CIFAR10Data():
    def __init__(self, data_path, device, validation_size = 0, is_one_hot = False):
        self.device = device
        self.num_train = 50000
        self.is_one_hot = is_one_hot
        self.validation_size = validation_size

        self.train_inputs, self.train_outputs = self.init_train(data_path)

        self.test_inputs, self.test_outputs, self.validation_inputs, self.validation_outputs = self.init_test_val(data_path)

    def init_train(self, data_path):
        self.training_data = datasets.CIFAR10(root=data_path,train=True,download=True,transform=ToTensor())
        train_inputs = torch.empty((50000,3,32,32))
        train_outputs = torch.empty((50000), dtype = torch.long)

        for idx, t in enumerate(self.training_data):
            data,label = t
            train_inputs[idx] = data
            train_outputs[idx] = label

        #One_hot
        if self.is_one_hot:
            train_outputs = F.one_hot(train_outputs, num_classes = 10).float()

        return train_inputs.to(self.device), train_outputs.to(self.device)
    
    def init_test_val(self, data_path):
        self.test_data = datasets.CIFAR10(root=data_path,train=False,download=True,transform=ToTensor())
        test_inputs = torch.empty((10000,3,32,32))
        test_outputs = torch.empty((10000), dtype = torch.long)

        for idx, t in enumerate(self.test_data):
            data,label = t
            test_inputs[idx] = data
            test_outputs[idx] = label

        #One_hot
        if self.is_one_hot:
            test_outputs = F.one_hot(test_outputs, num_classes = 10).float()
            return test_inputs[self.validation_size:,:,:,:].to(self.device), test_outputs[self.validation_size:,:].to(self.device), test_inputs[:self.validation_size,:,:,:].to(self.device), test_outputs[:self.validation_size,:].to(self.device)

        return test_inputs[self.validation_size:,:,:,:].to(self.device), test_outputs[self.validation_size:].to(self.device), test_inputs[:self.validation_size,:,:,:].to(self.device), test_outputs[:self.validation_size].to(self.device)

    def get_train_dataloader(self, batch_size = 32):
        return FastTensorDataLoader(self.train_inputs, self.train_outputs, batch_size = batch_size, shuffle = True)

    def get_test_dataloader(self, num_workers = 0, batch_size = 32):
        return FastTensorDataLoader(self.test_inputs, self.test_outputs, batch_size = batch_size, shuffle = True)

