import torch
import torch.autograd as autograd

""" 
In this file a lot of different SVGD implementations are collected, the basic structure of the class is the same of the 
standard SVGD.
"""

class SVGD:
    """
    Implementation of SVGD

    Args:
        P: instance of a distribution returning the log_prob, see distributions.py for examples
        K: kernel instance, see kernel.py for examples
        optimizer: instance of an optimizer SGD,Adam
    """
    def __init__(self, P, K, optimizer, temperature, ann_sch,num_train = False, noise = False):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.ann_schedule = ann_sch
        self.num_train = num_train
        self.noise = noise
        self.beta = temperature


    def phi(self, W,X,T,step):
        """
        Computes the update of the SVGD rule

        Args:
            W: particles
            X: inputs training batch
            T: labels training batch

        Return:
            phi: the update to feed the optimizer
            driving force: first term in the update rule
            repulsive force: second term in the update rule
        """
        
        if self.num_train: 
            num_t = self.P.num_train
        else: 
            num_t = 1
            
        W = W.detach().requires_grad_(True)
        
        #computing the driving force
        log_prob = self.P.log_prob(W,X,T)
        score_func = autograd.grad(log_prob.sum(), W)[0]
        
        #computing the repusive force
        K_W = self.K(W, W.detach())
        grad_K = -autograd.grad(K_W.sum(), W)[0]
        
        if self.noise: 
            lr = self.optim.state_dict()['param_groups'][0]['lr']
            K_W_exp =  torch.sqrt(2*K_W.repeat(W.shape[1],1,1)/(W.size(0)*lr))  #K_XX.repeat(X.shape[1],1,1)*2 
            langevin_noise = torch.randn_like(K_W_exp)*K_W_exp
            phi = ((1.0 / self.beta) * (self.ann_schedule[step]*K_W.detach().matmul(score_func)) +num_t*grad_K)/ W.size(0) + langevin_noise.sum(2).T
        else:
            phi = ((1.0 / self.beta) * (self.ann_schedule[step]*K_W.detach().matmul(score_func)) +num_t*grad_K) / W.size(0)


        return phi, self.ann_schedule[step]*K_W.detach().matmul(score_func), num_t*grad_K

    def step(self, W,X,T,step):
        """
        Customization of the optimizer step where I am forcing the gradients to be instead the SVGD update rule

        Args:
            W: particles
            X: input training batch
            T:  label training batch
        Return:
            driving force: first term in the update rule
            repulsive force: second term in the update rule
        """
        self.optim.zero_grad()
        update = self.phi(W,X,T,step)
        W.grad = -update[0]
        self.optim.step()
        return update[1], update[2]
