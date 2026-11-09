import torch

from kernel import RBF
from SVGD import SVGD
from WGD import WGD


def create_method(P,optimizer, inference, temperature, epochs, num_train, lr_halving, K=None, device = None):
    """
    Utils for the creation of the SVGD method
    """
    #Custom lr halving
    ann_sch = (0.5 ** (torch.arange(epochs) // lr_halving)).float()
    
    if K is None: 
        K = RBF()

    if inference == 'SVGD':
        K = RBF()
        method = SVGD(P,K,optimizer,temperature, ann_sch, num_train = num_train, noise = False)
    elif inference == 'WGD':
        K = RBF()
        method = WGD(P,K,optimizer, temperature, ann_sch, grad_estim=None, num_train = num_train, device = device)
    else:
        raise NotImplementedError("Inference Methods Implemented: SVGD & WGD")

    return method