import numpy as np
import torch

class Ensemble():
    """Implementation of an ensemble of models

    This is a simple class to manage and make predictions using an ensemble with or without particles
    Args:
        device: Torch device (cpu or gpu).
        net: pytorch model to create the ensemble
        particles(Tensor): Tensor (n_particles, n_params) containing squeezed parameter value of the specified model,
            if None, particles will be sample from a gaussian N(0,1)
        n_particles(int): if no particles are provided the ensemble is initialized and the number of members is required

    """
    def __init__(self, net, initial_particles, initial_buffers = None):
        self.net = net

        self.particles = initial_particles
        self.num_particles = self.particles.shape[0]
        self.particle_weighs_split = [int(np.prod(w)) for w in net.param_shapes]

        self.buffers = initial_buffers
        if not self.buffers == None:
            self.buffer_weighs_split = [int(np.prod(w)) for w in net.buffer_shapes]

    def reshape_model(self, z, weighs_split, shapes):
        reshaped_structure = []
        z_splitted = torch.split(z, weighs_split, 1)
        for j in range(z.shape[0]):
            l = []
            for i, shape in enumerate(shapes):
                l.append(z_splitted[i][j].reshape(shape))
            reshaped_structure.append(l)
        return reshaped_structure

    def forward(self, x, W=None, B=None):
        if W is None:
            W = self.particles
        if B is None:
            B = self.buffers

        models = self.reshape_model(W, self.particle_weighs_split, self.net.param_shapes)
        
        buffers = None
        if not self.buffers == None:
            buffers = self.reshape_model(B, self.buffer_weighs_split, self.net.buffer_shapes)
            pred,hidden = zip(*(list(self.net.forward(x,w,b)) for w, b in zip(models, buffers)))
        else:
            pred,hidden = zip(*(list(self.net.forward(x,w)) for w in models))

        return torch.stack(pred), torch.stack(hidden)
    