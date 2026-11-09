from laplace import Laplace
from laplace.curvature import AsdlGGN

import torch
from torch.nn import NLLLoss
import torch.nn.functional as F

from utils.util import ece

def calculateStudentTPriorPrecision(weights,std,df): 
    std = torch.Tensor([std]).cuda()
    df = torch.Tensor([df]).cuda()
    return torch.clamp((df+1)*(df*std**2-weights.pow(2))/(df*std**2+weights.pow(2)).pow(2), min=1e-8)

def get_predictive_method(la,data_loader, predictive_method, link_approx, device):
    targets = []
    probs = []
    with torch.no_grad():
        for (x,label)  in data_loader:
            x_test = x.to(device)
            label_test = label.to(device)
            predictionVec = la(x_test, pred_type = predictive_method, link_approx = link_approx, n_samples = 20)
        
            probs.append(predictionVec)
            targets.append(label_test)

    return torch.cat(targets),torch.cat(probs)

def getApproximation(model, device, train_loader, val_loader,predictive_method,link_approx,subset_of_weights,hessian_structure,
                     temperature,prior,std):

    
    if prior == "Gaussian":
        df = None
        std = torch.Tensor([float(std)])
        prior_precision = 1/std.pow(2)
    elif prior =="StudentT":
        df = 3

        weights = []
        if subset_of_weights=='last_layer':
            for param in model.fc.parameters():
                weights.append(param.flatten())
        else:
            for param in model.parameters():
                weights.append(param.flatten())
        weights = torch.cat(weights,dim=0)

        prior_precision = calculateStudentTPriorPrecision(weights,std,df)
    else:
        raise NotImplementedError("Priors Implemented: Gaussian & StudentT")
    
    la = Laplace(model, 'classification',
                 subset_of_weights=subset_of_weights,
                 hessian_structure=hessian_structure,
                 prior_precision = prior_precision,
                 temperature = temperature,
                 backend = AsdlGGN)
    
    la.fit(train_loader)

    targets, probs_laplace = get_predictive_method(la, val_loader, predictive_method, link_approx, device)

    targets = targets.cpu()
    probs_laplace = probs_laplace.cpu()

    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    ece_laplace = ece(targets.numpy(),probs_laplace.numpy())
    nll_laplace = NLLLoss()(torch.log(probs_laplace + 1e-20), targets)

    one_hot_label = F.one_hot(targets, num_classes = 10).float()
    rbs_laplace = ((probs_laplace-one_hot_label)**2).mean()**0.5
    
    return acc_laplace.item(), ece_laplace, nll_laplace.item(), rbs_laplace.item()