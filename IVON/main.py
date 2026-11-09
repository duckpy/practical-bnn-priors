import os 
os.chdir('..')

import math
import pickle
import numpy as np
import ivon

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.distributions import Laplace, StudentT, Cauchy, Normal

import sys
sys.path.insert(1, os.getcwd())

from models.standard import Base_CNN, ResNet18_CIFAR
from utils.generate_dataset import generate_dataset
from utils.util import get_all_experimental_parameters, ece


def train(net, dataloader_train, P, loss_fn, optimizer, train_samples, n_epochs, lr_halving, temperature, num_train, device):
    net.train()
    for epoch in tqdm(range(n_epochs)): 

        for (x, label) in dataloader_train: 

            optimizer.zero_grad() 
           
            train_x = x.to(device)
            train_label = label.to(device)

            for _ in range(train_samples):
                #Forward pass with Sampling
                with optimizer.sampled_params(train=True):
                    pred_vec = net(train_x)
                    loss_nll = loss_fn(pred_vec, train_label)

                    log_prior = sum(P.log_prob(p).sum() for p in net.parameters())
                    loss = loss_nll - log_prior / num_train

                    loss.backward()

                    optimizer.zero_grad()

            optimizer.step()    
          
        if epoch != 0 and epoch%lr_halving == 0: 
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5

def evaluate(net, optimizer, dataloader_test, batch_size, num_samples, device):
    net.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for (x, label) in dataloader_test:

            test_x = x.to(device)
            test_label = label.to(device)

            sampled_probs = []
            for _ in range(num_samples):
                with optimizer.sampled_params(train=False):
                    logits = net(test_x)
                    sampled_probs.append(F.softmax(logits, dim=1))

            prob_mean = torch.mean(torch.stack(sampled_probs), dim=0)

            all_probs.append(prob_mean.cpu())
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

def run_experiment(dataset, data, prior, temperature, lr, batch_size, lr_halving, train_samples, num_eval_samples, n_epochs, device):
    torch.manual_seed(44)
    np.random.seed(44)  

    #Distribution
    if prior == "Gaussian":
        P = Normal(loc = 0, scale = 1)
    elif prior == "StudentT":
        P = StudentT(df = 3, loc=0,scale = 1)
    elif prior == "Cauchy":
        P = Cauchy(loc=0,scale = 1)
    elif prior == "Laplace":
        P = Laplace(loc=0,scale = 1)
    else:
        raise NotImplementedError("Priors Implemented: Gaussian & StudentT")

    #Net
    if dataset == "CIFAR10":
        net = ResNet18_CIFAR().to(device)
    else: 
        net = Base_CNN().to(device)

    dataloader_train = data.get_train_dataloader(batch_size = batch_size)
    dataloader_test = data.get_test_dataloader(batch_size = batch_size)

    loss_fn = nn.CrossEntropyLoss()
    #Temperature applied through ESS
    optimizer = ivon.IVON(net.parameters(), lr = lr, ess = data.num_train / temperature, weight_decay = 0, beta1 = 0.9, hess_init = 0.05)

    train(net, dataloader_train, P, loss_fn, optimizer, train_samples, n_epochs, lr_halving, temperature, data.num_train, device)
    acc_test, nll_test, ece_test, rbs_test = evaluate(net, optimizer, dataloader_test, batch_size, num_eval_samples, device)
    
    return net, acc_test, nll_test, ece_test, rbs_test



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Output Folder
    if not os.path.exists(os.getcwd() + "/results"):
        os.makedirs(os.getcwd() + "/results")
    
    #Output Folder
    if not os.path.exists(os.getcwd() + "/results/IVON"):
        os.makedirs(os.getcwd() + "/results/IVON")

    #Parameters
    all_dataset, all_prior_type, all_temperature, all_lr, all_batch_size, all_lr_halving, n_epochs = get_all_experimental_parameters()

    train_samples = 10
    num_eval_samples = 20

    for dataset in all_dataset:
        data = generate_dataset(dataset, device)
        for prior in all_prior_type:
            for temperature in all_temperature:
                identifier = "IVON_" + dataset + "_" + prior + "_" + str(temperature)

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
                            print("Starting experiment:", dataset, prior, temperature, lr, batch_size, lr_halving)
                            start_time = time.time()
                            net, test_acc, test_nll, test_ece, test_rbs = run_experiment(dataset, data,
                                                                                    prior, temperature,
                                                                                    lr, batch_size,
                                                                                    lr_halving, train_samples, 
                                                                                    num_eval_samples, n_epochs, device)
                            end_time = time.time()
                            print(test_acc, test_nll, test_ece, test_rbs)
                            print("Ending experiment - Elapsed Time:", end_time - start_time)

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
                torch.save(best_net, os.getcwd() + "/results/IVON/" + identifier + ".pth")
                
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
                with open(os.getcwd() + "/results/IVON/"+ identifier +".pickle",'wb') as file:
                    pickle.dump(model_results, file)
                
                            