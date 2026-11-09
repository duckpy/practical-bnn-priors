import os
os.chdir('..')

import numpy as np
import pickle

import torch
import torch.nn.functional as F

from models.MFVI_models import BayesByBackprop_Base_CNN, BayesByBackprop_ResNet18_CIFAR
from utils.generate_dataset import generate_dataset
from utils.util import get_all_experimental_parameters, getAccuracy, ece

from tqdm import tqdm
import time

def train(net, n_epochs, num_samples, lr, lr_halving, dataloader_train, temp, device):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in tqdm(range(n_epochs)):
        for (x, label) in dataloader_train:
            optimizer.zero_grad()

            train_x = x.to(device)
            train_label = label.to(device)

            loss = 0

            for _ in range(num_samples):
                pred_vec = net(train_x)             
                log_posterior, log_prior = net.all_layer_KL()
                loglik_loss = F.nll_loss(F.log_softmax(pred_vec, dim=1),train_label)
                
                loss += 1/num_samples*(temp*log_posterior - log_prior + loglik_loss)
            
            loss.backward() 
            optimizer.step()

        if epoch%lr_halving == 0: 
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5    


def evaluate(net,dataloader_test,batch_size,num_samples,temp, device):
    with torch.no_grad():
    
        acc_test = getAccuracy(net, dataloader_test, batch_size, device)

        ELBO_test = 0
        kl_test = 0
        nll_test = 0
        ECE_test = 0
        mean_brier_test = 0

        for (x, label) in dataloader_test:

            test_x = x.to(device)
            test_label = label.to(device)

            kl = 0
            nll = 0
            ELBO = 0
            ECE = 0
            squared_error = 0

            for _ in range(0, num_samples):
                pred_vec = net(test_x)
                probs = F.softmax(pred_vec, dim = 1)    

                true_label_ece = test_label.cpu().detach().numpy()
                probs_ece = probs.cpu().detach().numpy()
                ece_val = ece(true_label_ece,probs_ece)

                log_posterior, log_prior = net.all_layer_KL()
                loglik_loss = F.nll_loss(F.log_softmax(pred_vec, dim=1),test_label)
                
                ELBO += float(1/num_samples*(temp*log_posterior - log_prior + loglik_loss).item())
                kl += float(1/num_samples*(log_posterior - log_prior).item()) 
                nll += float(1/num_samples*loglik_loss.item())
                ECE += float(1/num_samples*ece_val)

                one_hot_label = F.one_hot(test_label, num_classes=10).float()
                squared_error += float(1/num_samples*((probs-one_hot_label)**2).mean())

            ELBO_test += 1/len(dataloader_test) * ELBO
            kl_test += 1/len(dataloader_test) * kl
            nll_test += 1/len(dataloader_test) * nll
            ECE_test += 1/len(dataloader_test) * ECE

            mean_brier_test += 1/len(dataloader_test) * squared_error
        
        rbs = mean_brier_test**0.5
    
    return acc_test, ELBO_test, kl_test, nll_test, ECE_test, rbs


def run_experiment(dataset, data, prior, posterior, temperature, lr, batch_size, lr_halving, num_train_samples, num_eval_samples, n_epochs, device):
    torch.manual_seed(42)
    np.random.seed(42)

    if dataset == "CIFAR10":
        net = BayesByBackprop_ResNet18_CIFAR(prior, posterior, device).to(device)
    else: 
        net = BayesByBackprop_Base_CNN(prior, posterior, device).to(device)

    dataloader_train = data.get_train_dataloader(batch_size = batch_size)
    dataloader_test = data.get_test_dataloader(batch_size = batch_size)

    train(net, n_epochs, num_train_samples, lr, lr_halving, dataloader_train, temperature, device)

    acc_test, ELBO_test, kl_test, nll_test, ece_test, rbs_test = evaluate(net, dataloader_test, batch_size, num_eval_samples, temperature, device)

    return net, acc_test, nll_test, ece_test, rbs_test

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(43)
    np.random.seed(43)

    if not os.path.exists(os.getcwd() + "/results"):
        os.makedirs(os.getcwd() + "/results")
    
    if not os.path.exists(os.getcwd() + "/results/MFVI"):
        os.makedirs(os.getcwd() + "/results/MFVI")

    #Parameters
    all_dataset, all_prior_type, all_temperature, all_lr, all_batch_size, all_lr_halving, n_epochs = get_all_experimental_parameters()
    posterior = "Gaussian"

    num_train_samples = 10
    num_eval_samples = 10

    for dataset in all_dataset:
        data = generate_dataset(dataset, device)
        for prior in all_prior_type:
            for temperature in all_temperature:
                identifier = "MFVI_" + dataset + "_" + prior + "_" + str(temperature)

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
                                                                                         prior, posterior, 
                                                                                         temperature, lr, 
                                                                                         batch_size, lr_halving, 
                                                                                         num_train_samples, num_eval_samples,
                                                                                         n_epochs, device)
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
                torch.save(best_net, os.getcwd() + "/results/MFVI/" + identifier + ".pth")
                
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
                with open(os.getcwd() + "/results/MFVI/" + identifier + ".pickle",'wb') as file:
                    pickle.dump(model_results, file)


