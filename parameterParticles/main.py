import os
os.chdir('..')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle
import torch

import time

import sys
sys.path.insert(1, os.getcwd())

from utils.generate_dataset import generate_dataset
from utils.util import get_all_experimental_parameters
from training_ensemble import train_ensemble

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Output Folder
    if not os.path.exists(os.getcwd() + "/results"):
        os.makedirs(os.getcwd() + "/results")

    #Output Folder
    if not os.path.exists(os.getcwd() + "/results/SVGD"):
        os.makedirs(os.getcwd() + "/results/SVGD")

    #Parameters
    all_dataset, all_prior_type, all_temperature, all_lr, all_batch_size, all_lr_halving, n_epochs = get_all_experimental_parameters()
    inference = "SVGD" #WGD or SVGD

    for dataset in all_dataset:
        data = generate_dataset(dataset, device, is_one_hot = True)
        for prior_type in all_prior_type:
            for temperature in all_temperature:
                identifier = inference + "_" + dataset + "_" + prior_type + "_" + str(temperature)

                best_ensemble = None
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
                            torch.manual_seed(42)
                            np.random.seed(42)
                            print("Starting experiment:", inference, dataset, prior_type, temperature, lr, batch_size, lr_halving)
                            start_time = time.time()
                            ensemble, test_acc, test_nll, test_ece, test_rbs = train_ensemble(dataset, data, inference, lr, lr_halving, batch_size, temperature, prior_type, n_epochs, device)

                            end_time = time.time()
                            print(test_acc, test_nll, test_ece, test_rbs)
                            print("Ending experiment - Elapsed Time:", end_time - start_time)
                            
                            if ensemble == None or best_acc < test_acc:
                                best_acc = test_acc
                                best_nll = test_nll
                                best_ece = test_ece,
                                best_rbs = test_rbs
                                best_ensemble = ensemble
                                best_lr = lr
                                best_batch_size = batch_size
                                best_lr_halving = lr_halving

                #Save Model
                torch.save(best_ensemble, os.getcwd() + "/results/SVGD/" + identifier + ".pth")
                
                model_results = {
                    "acc": best_acc,
                    "nll": best_nll,
                    "ece": best_ece,
                    "rbs": best_rbs,
                    "lr": best_lr,
                    "batch_size": best_batch_size,
                    "lr_halving": best_lr_halving
                }
                
                #Saving Results
                with open(os.getcwd() + "/results/SVGD/" + identifier + ".pickle",'wb') as file:
                    pickle.dump(model_results, file)