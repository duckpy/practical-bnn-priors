import os 
os.chdir('..')
import numpy as np
import pickle
import time

import torch
import torch.nn.functional as F
from torch.distributions import StudentT, Normal

import sys
sys.path.insert(1, os.getcwd())

from laplace_approx import getApproximation
from models.standard import Base_CNN, ResNet18_CIFAR
from utils.generate_dataset import generate_dataset
from utils.util import get_all_experimental_parameters

from tqdm import tqdm

def train(net, P, n_epochs,lr,lr_halving,dataloader_train):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for epoch in tqdm(range(n_epochs)):  
        for (x, label) in dataloader_train:
            optimizer.zero_grad()  
           
            train_x = x.to(device)
            train_label = label.to(device)

            label_pred = net(train_x) 

            weights = [param.flatten() for param in net.parameters()]
            weights = torch.cat(weights,dim=0)

            log_prior = P.log_prob(weights).mean()
            NLL = F.cross_entropy(label_pred,train_label)
            loss = NLL-log_prior
            
            loss.backward() 
            optimizer.step()    

        if epoch%lr_halving == 0: 
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5

def run_SGD_pretrain(dataset, data, prior, lr, batch_size, lr_halving, device):
    torch.manual_seed(42)
    np.random.seed(42)

    if prior == "Gaussian":
        P = Normal(loc = 0, scale = 1)
    elif prior == "StudentT":
        P = StudentT(df = 3 , loc = 0 , scale = 1)
    else:
        raise NotImplementedError("Priors Implemented: Gaussian & StudentT")
    
    if dataset == "CIFAR10":
        net = ResNet18_CIFAR().to(device)
    else: 
        net = Base_CNN().to(device)

    dataloader_train = data.get_train_dataloader(batch_size = batch_size)

    train(net, P, n_epochs, lr, lr_halving, dataloader_train)

    return net

def run_laplace_approximation(net, data, batch_size, predictive_method, link_approx, subset_of_weights, hessian_structure, temperature, prior, device, prior_std = 1):

    dataloader_train = data.get_train_dataloader(batch_size = batch_size)
    dataloader_test = data.get_test_dataloader(batch_size = batch_size)
    
    acc_laplace, ece_laplace, nll_laplace, rbs_laplace = getApproximation(net, device,
                                                                        dataloader_train, 
                                                                        dataloader_test,
                                                                        predictive_method,
                                                                        link_approx,
                                                                        subset_of_weights,
                                                                        hessian_structure,
                                                                        temperature,prior,
                                                                        prior_std)
    
    return acc_laplace, ece_laplace, nll_laplace, rbs_laplace


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(os.getcwd() + "/results"):
        os.makedirs(os.getcwd() + "/results")
    
    if not os.path.exists(os.getcwd() + "/results/Laplace"):
        os.makedirs(os.getcwd() + "/results/Laplace")

    #Parameters
    all_dataset, all_prior_type, all_temperature, all_lr, all_batch_size, all_lr_halving, n_epochs = get_all_experimental_parameters(is_laplace = True)
    predictive_method = 'glm'
    link_approx = 'mc'              
    subset_of_weights = 'last_layer'   
    hessian_structure = 'diag'     

    for dataset in all_dataset:
        data = generate_dataset(dataset, device)
        for prior in all_prior_type:
            for temperature in all_temperature:
                identifier = "Laplace_" + predictive_method + "_" + dataset + "_" + prior + "_" + str(temperature)

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
                            print("Starting experiment:", dataset, predictive_method, prior, temperature, lr, batch_size, lr_halving)
                            start_time = time.time()
                            net = run_SGD_pretrain(dataset, data, prior, lr, batch_size, lr_halving, device)
                            laplace_test_acc, laplace_test_nll, laplace_test_ece, laplace_test_rbs = run_laplace_approximation(net, data, 
                                                                                                                                batch_size, predictive_method, 
                                                                                                                                link_approx, subset_of_weights, 
                                                                                                                                hessian_structure, temperature, 
                                                                                                                                prior, device)
                            end_time = time.time()
                            print(laplace_test_acc, laplace_test_nll, laplace_test_ece, laplace_test_rbs)
                            print("Ending experiment - Elapsed Time:", end_time - start_time)
                            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                            
                            if best_net == None or best_acc < laplace_test_acc:
                                best_acc = laplace_test_acc
                                best_nll = laplace_test_nll
                                best_ece = laplace_test_ece,
                                best_rbs = laplace_test_rbs
                                best_net = net.state_dict()
                                best_lr = lr
                                best_batch_size = batch_size
                                best_lr_halving = lr_halving

                #Save Model
                torch.save(best_net, os.getcwd() + "/results/Laplace/" + identifier + ".pth")
                
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
                with open(os.getcwd() + "/results/Laplace/" + identifier + ".pickle", 'wb') as file:
                    pickle.dump(model_results, file)