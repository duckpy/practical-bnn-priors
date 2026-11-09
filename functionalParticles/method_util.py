import torch

from utils.SSGE_squeeze import SpectralSteinEstimator
from kernel import RBF
from f_SVGD import f_SVGD
from f_WGD import f_WGD


def create_method(P,optimizer, inference, temperature, epochs, num_train, lr_halving, K=None, device = None):
    """
    Utils for the creation of the SVGD method
    """
    #Custom lr halving
    ann_sch = (0.5 ** (torch.arange(epochs) // lr_halving)).float()
    
    if K is None: 
        K = RBF()
    
    if inference == 'f_SVGD':
        K = RBF()
        ssge = SpectralSteinEstimator(0.01,None,K, device = device)
        method = f_SVGD(P, K, optimizer,ssge,temperature, ann_sch, num_train = num_train)
    elif inference == 'f_WGD':
        K = RBF()
        ssge = SpectralSteinEstimator(0.01,None,K, device = device)
        method = f_WGD(P, K, optimizer, ssge, temperature, ann_sch, num_train = num_train)
    else:
        raise NotImplementedError("Inference Methods Implemented: SVGD & WGD")

    return method