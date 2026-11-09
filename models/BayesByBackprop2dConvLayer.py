import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.distributions import Laplace,StudentT,Cauchy,Normal

def getPriorDistrConv(prior,priorMean_init,priorStd_init,out_channels,flattendDim, device):
    W_var_prior = torch.Tensor([priorStd_init]).to(device)#/ self.indim]) #/ np.sqrt(self.indim)])
    W_var_prior = W_var_prior.repeat(flattendDim)
    W_mean_prior = torch.Tensor([priorMean_init]).to(device)
    W_mean_prior = W_mean_prior.repeat(flattendDim)

    b_var_prior = torch.Tensor([priorStd_init]).to(device) # / np.sqrt(self.indim)]) # was 1 
    b_var_prior = b_var_prior.repeat(out_channels)
    b_mean_prior = torch.Tensor([priorMean_init]).to(device)
    b_mean_prior = b_mean_prior.repeat(out_channels)

    # Prior distribution over weights and bias (will NOT be changed)
    if prior == "Laplace":
        priorDistrWeights = Laplace(loc = W_mean_prior, scale = W_var_prior)   
        priorDistrBias = Laplace(loc = b_mean_prior, scale = b_var_prior)

    elif prior == "StudentT":
        priorDistrWeights = StudentT(df = 3,loc = W_mean_prior, scale = W_var_prior)
        priorDistrBias = StudentT(df = 3,loc = b_mean_prior, scale = b_var_prior)

    elif prior == "Cauchy":
        priorDistrWeights = Cauchy(loc = W_mean_prior, scale = W_var_prior)
        priorDistrBias = Cauchy(loc = b_mean_prior, scale = b_var_prior)

    elif prior == "Gaussian":
        priorDistrWeights = Normal(loc = W_mean_prior, scale = W_var_prior)
        priorDistrBias = Normal(loc = b_mean_prior, scale = b_var_prior)

    else: 
        raise NotImplementedError("Priors Implemented: Gaussian, Laplace, Cauchy & StudentT")
    
    return priorDistrWeights, priorDistrBias



def getStandPosteriorDistrConv(posterior,flattendDim,out_channels,device):
    # Standard loc-scale distributions for reparametrization trick in posterior distrbution
    if posterior == "Gaussian":
        postStdDistrWeights = Normal(loc=torch.zeros(flattendDim).to(device), 
                                 scale=torch.ones(flattendDim).to(device))
        postStdDistrBias = Normal(loc=torch.zeros(out_channels).to(device), 
                                 scale=torch.ones(out_channels).to(device))

    elif posterior == "Laplace": 
        postStdDistrWeights = Laplace(loc=torch.zeros(flattendDim).to(device), 
                                     scale=torch.ones(flattendDim).to(device))
        postStdDistrBias = Laplace(loc=torch.zeros(out_channels).to(device), 
                                 scale=torch.ones(out_channels).to(device))

    elif posterior == "Cauchy": 
        postStdDistrWeights = Cauchy(loc=torch.zeros(flattendDim).to(device), 
                                     scale=torch.ones(flattendDim).to(device))
        postStdDistrBias = Cauchy(loc=torch.zeros(out_channels).to(device), 
                                 scale=torch.ones(out_channels).to(device))

    elif posterior == "StudentT": 
        postStdDistrWeights = StudentT(df = 3,loc=torch.zeros(flattendDim).to(device), 
                                     scale=torch.ones(flattendDim).to(device))
        postStdDistrBias = StudentT(df = 3,loc=torch.zeros(out_channels).to(device), 
                                 scale=torch.ones(out_channels).to(device))
    else: 
        raise NotImplementedError("Posteriors Implemented: Gaussian, Laplace, Cauchy & StudentT")
        
        
    return postStdDistrWeights, postStdDistrBias



class BayesByBackprop2DConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,prior,posterior, device, stride = 1, padding = 0):
        super(BayesByBackprop2DConvLayer,self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prior = prior
        self.posterior = posterior
        self.kernel_size = kernel_size
        self.flattendDim = self.out_channels * self.in_channels * self.kernel_size * self.kernel_size

        self.stride = stride
        self.padding = padding
        
        posterior_sigma_init = 0.001 # 0.5 
        posterior_rho_init= -10 # to have ~zero mean init
        
        priorMean_init = 0.0
        priorStd_init = 1.0
        
        # Keeping track of current weight samples 
        self.currentWeightsSample = None 
        self.currentBiasSample = None 
        
        # POSTERIOR: Initializing the posterior =================================================
        # Variational weight (W) and bias (b) parameters
        
        self.post_w_mean = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size,self.kernel_size).to(device))
        init.kaiming_uniform_(self.post_w_mean,mode='fan_out',nonlinearity='relu')   # He initialitation
  
        self.post_b_mean = nn.Parameter(torch.Tensor(self.out_channels).to(device))
        init.zeros_(self.post_b_mean)
 
        self.post_W_rho = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size,self.kernel_size).to(device))
        self.post_W_rho.data.normal_(posterior_rho_init, posterior_sigma_init) 

        self.post_b_rho = nn.Parameter(torch.Tensor(self.out_channels).to(device))
        self.post_b_rho.data.normal_(posterior_rho_init, posterior_sigma_init)
   
        # Note: He & Glorot initilialization of rho param of weights and bias did not produce meaningful results
        
        # Initializing Standarf Posteiror Distribution (loc = 0, scale = 1)
        self.postStdDistrWeights, self.postStdDistrBias = getStandPosteriorDistrConv(self.posterior,
                                                                                     self.flattendDim,
                                                                                     self.out_channels, device)
                                                                                                                                                               
        # PRIOR: Initializing the prior ========================================================       
        self.priorDistrWeights,self.priorDistrBias = getPriorDistrConv(self.prior,priorMean_init,
                                                                            priorStd_init,self.out_channels,
                                                                            self.flattendDim, device)
                
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
        postStdSampleWeights = self.postStdDistrWeights.rsample(sample_shape=torch.Size([1]))
        mu = self.post_w_mean.view(self.flattendDim)
        scale = torch.log(1+torch.exp(self.post_W_rho.view(self.flattendDim)))
        postSampleWeights = mu + scale * postStdSampleWeights # reparametrization trick for loc-sclae distributions
        posterior_sample_matrix_weights = torch.reshape(postSampleWeights,
                                                        (self.out_channels, self.in_channels, 
                                                         self.kernel_size,self.kernel_size))
        
        # Bias term
        postStdSampleBias = self.postStdDistrBias.rsample(sample_shape=torch.Size([1])).squeeze()
        mu = self.post_b_mean 
        scale = torch.log(1+torch.exp(self.post_b_rho))
        posterior_sample_bias = mu + scale * postStdSampleBias # reparametrization trick
         
        return posterior_sample_matrix_weights, posterior_sample_bias  
    
    def KL_divergence(self):
        """
        Calculates the KL divergence KL(q(w|D)||P(w)) using n samples from posterior for weight and bias term. 
        Taking the sum of log-posterior and log-prior over samples.
        The sampling is called in every forward step [forward(self,x)] and based on these samples 
        the kl-divergence is calculated. 
        """
        currentWeightsSample = torch.reshape(self.currentWeightsSample, (1,self.flattendDim))
        
        # Posterior q(w|D) - summing over variational posterior for weights and bias 
        if self.posterior == "Gaussian":
            self.posteriorDistribution_w = Normal(loc=self.post_w_mean.view(self.flattendDim), 
                                                  scale=torch.log(1+torch.exp(self.post_W_rho.view(self.flattendDim)))) 
            self.posteriorDistribution_b = Normal(loc=self.post_b_mean.view(self.out_channels), 
                                                  scale=torch.log(1+torch.exp(self.post_b_rho.view(self.out_channels))))

        elif self.posterior == "Laplace":
            self.posteriorDistribution_w = Laplace(loc=self.post_w_mean.view(self.flattendDim), 
                                         scale=torch.log(1+torch.exp(self.post_W_rho.view(self.flattendDim)))) 
            self.posteriorDistribution_b = Laplace(loc=self.post_b_mean.view(self.out_channels), 
                                                  scale=torch.log(1+torch.exp(self.post_b_rho.view(self.out_channels))))
     
        elif self.posterior == "Cauchy":
            self.posteriorDistribution_w = Cauchy(loc=self.post_w_mean.view(self.flattendDim), 
                                         scale=torch.log(1+torch.exp(self.post_W_rho.view(self.flattendDim))))    
            self.posteriorDistribution_b = Cauchy(loc=self.post_b_mean.view(self.out_channels), 
                                                  scale=torch.log(1+torch.exp(self.post_b_rho.view(self.out_channels))))
        
        elif self.posterior == "StudentT":
            self.posteriorDistribution_w = StudentT(df = 3,loc=self.post_w_mean.view(self.flattendDim), 
                                         scale=torch.log(1+torch.exp(self.post_W_rho.view(self.flattendDim)))) 
            self.posteriorDistribution_b = StudentT(df = 3,loc=self.post_b_mean.view(self.out_channels), 
                                                  scale=torch.log(1+torch.exp(self.post_b_rho.view(self.out_channels))))
        else:
            raise NotImplementedError("Posteriors Implemented: Gaussian, Laplace, Cauchy & StudentT")

            
        log_posterior_weights = self.posteriorDistribution_w.log_prob(currentWeightsSample).mean()
        log_posterior_bias = self.posteriorDistribution_b.log_prob(self.currentBiasSample).mean()
        log_posterior = log_posterior_weights + log_posterior_bias
        
        # Prior P(w) - summing over prior for weights and bias
        log_prior_weights = self.priorDistrWeights.log_prob(currentWeightsSample).mean() 
        log_prior_bias = self.priorDistrBias.log_prob(self.currentBiasSample).mean()
        log_prior = log_prior_weights + log_prior_bias
        
        #return log_posterior - log_prior 
        return log_posterior, log_prior
    
    def forward(self, x):
        """
        Sampling from variational posterior q(w|D) in the forward step.
        """
        self.currentWeightsSample, self.currentBiasSample = self.sample_posterior()

        return F.conv2d(x, self.currentWeightsSample, self.currentBiasSample, stride = self.stride, padding=self.padding)
    
    
    