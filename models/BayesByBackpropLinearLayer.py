import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.distributions import Laplace,StudentT,Cauchy,Normal

def getPriorDistr(prior,priorMean_init,priorStd_init,indim,outdim,device):
    W_var_prior = torch.Tensor([priorStd_init]).to(device) 
    W_var_prior = W_var_prior.repeat(indim*outdim)
    W_mean_prior = torch.Tensor([priorMean_init]).to(device)
    W_mean_prior = W_mean_prior.repeat(indim*outdim)

    b_var_prior = torch.Tensor([priorStd_init]).to(device)
    b_var_prior = b_var_prior.repeat(outdim)
    b_mean_prior = torch.Tensor([priorMean_init]).to(device)
    b_mean_prior = b_mean_prior.repeat(outdim)

    # Prior distribution over weights and bias (will NOT be changed)
    if prior == "Gaussian":
        priorDistrWeights = Normal(loc = W_mean_prior, scale = W_var_prior)
        priorDistrBias = Normal(loc=b_mean_prior, scale=b_var_prior)

    elif prior == "Laplace":
        priorDistrWeights = Laplace(loc = W_mean_prior.view(indim*outdim),
                                    scale = W_var_prior.view(indim*outdim))
        priorDistrBias = Laplace(loc=b_mean_prior,
                                      scale=b_var_prior)            
    elif prior == "StudentT":
        priorDistrWeights = StudentT(df = 3,loc = W_mean_prior, scale = W_var_prior)
        priorDistrBias = StudentT(df = 3,loc=b_mean_prior, scale= b_var_prior)

    elif prior == "Cauchy":
        priorDistrWeights = Cauchy(loc = W_mean_prior, scale = W_var_prior)
        priorDistrBias = Cauchy(loc=b_mean_prior, scale=b_var_prior)

    else: 
        raise NotImplementedError("Priors Implemented: Gaussian, Laplace, Cauchy & StudentT")

    return priorDistrWeights, priorDistrBias



def getStandPosteriorDistr(posterior, indim, outdim, device):
    if posterior == "Gaussian":
        postStdDistrWeights = Normal(loc=torch.zeros(outdim * indim).to(device), 
                                     scale=torch.ones(outdim * indim).to(device))
        postStdDistrBias = Normal(loc=torch.zeros(outdim).to(device), 
                                     scale=torch.ones(outdim).to(device)) 
    elif posterior == "Laplace": 
        postStdDistrWeights = Laplace(loc=torch.zeros(outdim * indim).to(device), 
                                     scale=torch.ones(outdim * indim).to(device))
        postStdDistrBias = Laplace(loc=torch.zeros(outdim).to(device), 
                                     scale=torch.ones(outdim).to(device))
    elif posterior == "Cauchy": 
        postStdDistrWeights = Cauchy(loc=torch.zeros(outdim * indim).to(device), 
                                     scale=torch.ones(outdim * indim).to(device))
        postStdDistrBias = Cauchy(loc=torch.zeros(outdim).to(device), 
                                     scale=torch.ones(outdim).to(device))
    elif posterior == "StudentT": 
        postStdDistrWeights = StudentT(df = 3,loc=torch.zeros(outdim * indim).to(device), 
                                     scale=torch.ones(outdim * indim).to(device))
        postStdDistrBias = StudentT(df = 3,loc=torch.zeros(outdim).to(device), 
                                     scale=torch.ones(outdim).to(device))
    else: 
        raise NotImplementedError("Posteriors Implemented: Gaussian, Laplace, Cauchy & StudentT")
        
    return postStdDistrWeights, postStdDistrBias



class BayesByBackpropLayer(nn.Module):
    def __init__(self, indim, outdim,prior,posterior,device):
        super(BayesByBackpropLayer,self).__init__()
        self.device = device
        self.indim = indim
        self.outdim = outdim
        self.prior = prior
        self.posterior = posterior

        # as in "Multiplicative Normalizing Flows for Variational Bayesian Neural Networks", Louizos et al. 2017
        posterior_sigma_init =  0.001 #0.5
        posterior_rho_init= -10 # to ~have zero mean init
        
        priorMean_init = 0.0
        priorStd_init = 1.0
        
        # Keeping track of current weight samples 
        self.currentWeightsSample = None 
        self.currentBiasSample = None 
        
        # POSTERIOR: Initializing the posterior ======================================================== 
        # Variational weight (W) and bias (b) parameters
        self.post_w_mean = nn.Parameter(torch.Tensor(self.outdim,self.indim))
        init.kaiming_uniform_(self.post_w_mean,mode='fan_out',nonlinearity='relu')  # He initialitation
        
        self.post_b_mean = nn.Parameter(torch.Tensor(self.outdim,1).to(device))
        init.zeros_(self.post_b_mean)

        self.post_W_rho = nn.Parameter(torch.Tensor(self.outdim,self.indim).to(device))
        self.post_W_rho.data.normal_(posterior_rho_init, posterior_sigma_init)
       
        self.post_b_rho = nn.Parameter(torch.Tensor(self.outdim,1).to(device))
        self.post_b_rho.data.normal_(posterior_rho_init, posterior_sigma_init)
        
        # Standard loc-scale distributions for reparametrization trick in posterior distrbution
        self.postStdDistrWeights, self.postStdDistrBias = getStandPosteriorDistr(self.posterior, 
                                                                                 self.indim, self.outdim, device)
        
        # PRIOR: Initializing the prior ========================================================    
        self.priorDistrWeights, self.priorDistrBias = getPriorDistr(self.prior,
                                                                    priorMean_init,priorStd_init,
                                                                    self.indim,self.outdim, device)    
        
    def sample_posterior(self):
        """
        Samples n times from the posterior of the NN weights w and the bias using the local 
        reparametrization trick.
        :return: 2 torch tensors:
        .(0) tensor of shape n x indim x outdim, holding the weight samples from the posterior of the weights
        .(1) tensor of shape n x outdim, holding the bias samples from the posterior of the bias term
        """
        
        # Local reparametrization trick 
        # Weights
        postStdSampleWeights = self.postStdDistrWeights.rsample(sample_shape=torch.Size([1])) # sample_shape=torch.Size([n])
        mu = self.post_w_mean.view(self.indim*self.outdim)
        scale = torch.log(1+torch.exp(self.post_W_rho.view(self.indim*self.outdim)))
        postSampleWeights = mu + scale * postStdSampleWeights # reparametrization trick loc-scale distr.
        posterior_sample_matrix_weights = torch.reshape(postSampleWeights,(1,self.outdim,self.indim))
        
        # Bias term
        postStdSampleBias = self.postStdDistrBias.rsample(sample_shape=torch.Size([1]))
        postStdSampleBias = torch.reshape(postStdSampleBias, (self.outdim,1))
        mu = self.post_b_mean
        scale = torch.log(1+torch.exp(self.post_b_rho))
        posterior_sample_bias = mu + scale * postStdSampleBias
        posterior_sample_bias = torch.reshape(posterior_sample_bias, (1,self.outdim))
        
        return posterior_sample_matrix_weights, posterior_sample_bias 
    
    def KL_divergence(self):
        """
        Calculates the KL divergence KL(q(w|D)||P(w)) using n samples from posterior for weight and bias term. 
        Taking the sum of log-posterior and log-prior over samples.
        The sampling is called in every forward step [forward(self,x)] and based on these samples 
        the kl-divergence is calculated. 
        """
        currentWeightsSample = torch.reshape(self.currentWeightsSample, (1,self.indim*self.outdim))
        
        # Prior P(w) - summing over prior for weights and bias
        log_prior = self.priorDistrWeights.log_prob(currentWeightsSample).mean() + \
        self.priorDistrBias.log_prob(self.currentBiasSample).mean()
                
        # Posterior q(w|D) - summing over variational posterior for weights and bias         
        if self.posterior == "Laplace":
            self.posteriorDistribution_w = Laplace(loc=self.post_w_mean.view(self.indim*self.outdim), 
                                         scale=torch.log(1+torch.exp(self.post_W_rho.view(self.indim*self.outdim)))) 
            self.posteriorDistribution_b = Laplace(loc=self.post_b_mean.view(self.outdim), 
                                         scale=torch.log(1+torch.exp(self.post_b_rho.view(self.outdim))))
            
        elif self.posterior == "Cauchy":
            self.posteriorDistribution_w = Cauchy(loc=self.post_w_mean.view(self.indim*self.outdim), 
                                         scale=torch.log(1+torch.exp(self.post_W_rho.view(self.indim*self.outdim)))) 
            self.posteriorDistribution_b = Cauchy(loc=self.post_b_mean.view(self.outdim), 
                                         scale=torch.log(1+torch.exp(self.post_b_rho.view(self.outdim))))
            
        elif self.posterior == "StudentT":
            self.posteriorDistribution_w = StudentT(df = 3,loc=self.post_w_mean.view(self.indim*self.outdim), 
                                         scale=torch.log(1+torch.exp(self.post_W_rho.view(self.indim*self.outdim)))) 
            self.posteriorDistribution_b = StudentT(df = 3,loc=self.post_b_mean.view(self.outdim), 
                                         scale=torch.log(1+torch.exp(self.post_b_rho.view(self.outdim))))
        
        elif self.posterior == "Gaussian":
            self.posteriorDistribution_w = Normal(loc=self.post_w_mean.view(self.indim*self.outdim), 
                                         scale=torch.log(1+torch.exp(self.post_W_rho.view(self.indim*self.outdim))))
            self.posteriorDistribution_b = Normal(loc=self.post_b_mean.view(self.outdim), 
                                         scale=torch.log(1+torch.exp(self.post_b_rho.view(self.outdim))))    
        else:
            raise NotImplementedError("Posteriors Implemented: Gaussian, Laplace, Cauchy & StudentT")

            
        log_posterior = self.posteriorDistribution_w.log_prob(currentWeightsSample).mean() + \
        self.posteriorDistribution_b.log_prob(self.currentBiasSample).mean()
        
        return log_posterior, log_prior
    
    def forward(self, x):
        """
        Sampling from variational posterior q(w|D) in the forward step.
        """
        self.currentWeightsSample,self.currentBiasSample = self.sample_posterior()

        return F.linear(input = x, 
                        weight = self.currentWeightsSample.squeeze(), 
                        bias = self.currentBiasSample.squeeze())
    
    