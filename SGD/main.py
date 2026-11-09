import os 
os.chdir('..')

import time

import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(1, os.getcwd())

from models.standard import Base_CNN, ResNet18_CIFAR
from utils.generate_dataset import generate_dataset
from utils.util import get_all_experimental_parameters, ece

def train(net, dataloader_train, loss_fn, optimizer, n_epochs, lr_halving, device):
    net.train()
    for epoch in range(n_epochs): 

        for (x, label) in dataloader_train: 
            optimizer.zero_grad() 
           
            train_x = x.to(device)
            train_label = label.to(device)

            pred_vec = net(train_x)
            loss = loss_fn(pred_vec, train_label)
            loss.backward()

            optimizer.step()   
          
        if epoch != 0 and epoch%lr_halving == 0: 
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5

def evaluate(net, dataloader_test, batch_size):
    net.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for (x, label) in dataloader_test:
            test_x = x.to(device)
            test_label = label.to(device)
            pred_vec = net(test_x)
            probs = F.softmax(pred_vec, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(test_label.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    preds = torch.argmax(all_probs, dim=1)

    #Accuracy
    acc_test = sum(preds == all_labels) / (len(dataloader_test)*batch_size)

    #NLL
    nll_test = F.nll_loss(all_probs.log(), all_labels).item()
            
    #ECE
    true_label_ece = all_labels.detach().numpy()
    probs_ece = all_probs.detach().numpy()
    ece_test = ece(true_label_ece,probs_ece)

    #RBS
    one_hot_label = F.one_hot(all_labels, num_classes = 10).float()
    rbs_test = ((all_probs-one_hot_label)**2).mean().sqrt().item()
            
    return acc_test.item(), nll_test, ece_test, rbs_test

def run_experiment(dataset, data, lr, batch_size, lr_halving, n_epochs, device):
    torch.manual_seed(42)
    np.random.seed(42)  

    #Net
    if dataset == "CIFAR10":
        net = ResNet18_CIFAR().to(device)
    else: 
        net = Base_CNN().to(device)

    #Dataloaders
    dataloader_train = data.get_train_dataloader(batch_size = batch_size)
    dataloader_test = data.get_test_dataloader(batch_size = batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = lr, momentum = 0.9)

    train(net, dataloader_train, loss_fn, optimizer, n_epochs, lr_halving, device)
    acc_test, nll_test, ece_test, rbs_test = evaluate(net, dataloader_test, batch_size)

    return net, acc_test, nll_test, ece_test, rbs_test



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Output Folder
    if not os.path.exists(os.getcwd() + "/results"):
        os.makedirs(os.getcwd() + "/results")
    
    #Output Folder
    if not os.path.exists(os.getcwd() + "/results/SGD"):
        os.makedirs(os.getcwd() + "/results/SGD")

    #Parameters
    all_dataset, all_prior_type, all_temperature, all_lr, all_batch_size, all_lr_halving, n_epochs = get_all_experimental_parameters()

    for dataset in all_dataset:
        data = generate_dataset(dataset)
        identifier = "SGD_" + dataset 

        best_net = None
        best_acc = 0
        best_nll = 0
        best_ece = 0
        best_rbs = 0
        best_lr = None
        best_batch_size = None
        best_lr_halving = None

        #Hyperparameters
        for lr in all_lr:
            for batch_size in all_batch_size:
                for lr_halving in all_lr_halving:
                    print("Starting experiment:", dataset, lr, batch_size, lr_halving)
                    start_time = time.time()
                    net, test_acc, test_nll, test_ece, test_rbs = run_experiment(dataset, data,
                                                                            lr, batch_size,
                                                                            lr_halving, n_epochs, device)
                    end_time = time.time()
                    print("Ending Experiment - Elapsed Time:", end_time - start_time)
                    
                    if best_net == None or best_acc < test_acc:
                        best_acc = test_acc
                        best_nll = test_nll
                        best_ece = test_ece,
                        best_rbs = test_rbs
                        best_net = net.state_dict()
                        best_lr = lr
                        best_batch_size = batch_size
                        best_lr_halving = lr_halving

        #Save Model
        torch.save(best_net, os.getcwd() + "/results/SGD/" + identifier + ".pth")

        model_results = {
            "acc": best_acc,
            "nll": best_nll,
            "ece": best_ece,
            "rbs": best_rbs,
            "lr": best_lr,
            "batch_size": best_batch_size,
            "lr_halving": best_lr_halving
        }

        #Save Results
        with open(os.getcwd() + "/results/SGD/"+ identifier +".pickle",'wb') as file:
            pickle.dump(model_results, file)

                
                            