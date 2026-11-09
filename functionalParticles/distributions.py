import torch
import torch.nn.functional as F

class Unorm_post():
    """
    Implementation of unnormalized posterior for a neural network model. It assume gaussian likelihood with variance
    config.pred_dist_std. The prior can be freely specified, the only requirement is a log_prob method to return the
    log probability of the particles.

    Args:
        ensemble: ensemble instance from MLP.py
        prior: prior instance from torch.distributions or custom, .log_prob() method is required
        config: Command-line arguments.
        n_train: number of training datapoints to rescale the likelihood

    """
    def __init__(self, ensemble, prior, n_train,add_prior = True):
        self.prior = prior
        self.ensemble = ensemble
        self.num_train = n_train
        self.add_prior = add_prior


    def log_prob(self, particles, X, T, return_loss=False, return_pred = False, pred_idx = 1):
        pred = self.ensemble.forward(X, particles)

        if self.ensemble.net.classification:
            if pred_idx == 1:
                loss = torch.stack([F.nll_loss(F.log_softmax(p), T.argmax(1)) for p in pred[1]])
            else:
                loss = (-(T.expand_as(pred[1])*torch.log(pred[0]+1e-15))).max(2)[0].sum(1)/X.shape[0]

        else:
            loss = 0.5*torch.mean((T.expand_as(pred[0])-pred[0])**2,1)


        ll = -loss*self.num_train

        if particles is None:
            particles = self.ensemble.particles

        if self.add_prior:
            log_prob = torch.add(self.prior.log_prob(particles).sum(1), ll)
        else:
            log_prob = ll
        if return_loss:
            return torch.mean(loss),pred[0]
        elif return_pred:
            return log_prob,pred #0 softmax, 1 is logit
        else:
            return log_prob