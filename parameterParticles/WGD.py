import torch.autograd as autograd

class WGD:
    """
    Implementation of WGD in weight space with KDE, SGE, SSGE approximation 

    Args:
        P: instance of a distribution returning the log_prob, see distributions.py for examples
        K: kernel instance, see kernel.py for examples
        optimizer: instance of an optimizer SGD,Adam
    """
    def __init__(self, P, K, optimizer, temperature,ann_sch,grad_estim=None,num_train = False, device = None):
        self.P = P
        self.K = K
        self.optim = optimizer
        self.pge = grad_estim
        self.ann_schedule=ann_sch
        self.num_train = num_train
        self.device = device
        self.beta = temperature


    def phi(self, W,X,T,step):
        """
        Computes the update of the WGD rule

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

        #kde
        K_W = self.K(W, W.detach())
        grad_K = autograd.grad(K_W.sum(), W)[0] 
        grad_density = grad_K/ K_W.sum(1,keepdim = True)
        
        phi = (1.0 / self.beta) * (self.ann_schedule[step]*score_func)-grad_density

        return phi, self.ann_schedule[step]*score_func, grad_density

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