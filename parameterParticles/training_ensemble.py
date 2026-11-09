import torch
import torch.nn.functional as F

from distributions import Unorm_post
from ensemble import Ensemble
from method_util import create_method

from utils.util import ece
from models.state_models import Base_CNN, ResNet18_CIFAR

from tqdm import tqdm

def train_ensemble(dataset, data, inference, lr, lr_halving, batch_size, temperature, prior_type, epochs, device, n_particles = 10):

    if dataset == "CIFAR10": 
        net = ResNet18_CIFAR(act=F.relu, out_act = F.softmax).to(device)
        model = ResNet18_CIFAR
        has_buffer = True
    else: 
        net = Base_CNN(act=F.relu, out_act = F.softmax).to(device)
        model = Base_CNN
        has_buffer = False


    #Init Particles
    l = []
    for _ in range(n_particles):
        l.append(torch.cat([p.flatten() for p in model(act=F.relu, out_act = F.softmax).parameters()]).detach())
    init_particles = torch.stack(l).to(device)

    #Init Buffers
    init_buffers = None
    if has_buffer:
        k = []
        for _ in range(n_particles):
            k.append(torch.cat([b.flatten() for b in model(act=F.relu, out_act = F.softmax).buffers()]).detach())
        init_buffers = torch.stack(k).to(device)

    ensemble = Ensemble(net=net, initial_particles = init_particles, initial_buffers = init_buffers)

    #Setting utils and distributions
    W = ensemble.particles

    optimizer = torch.optim.Adam([W], lr, weight_decay=0,
                                 betas=[0.9, 0.999])
    
    if prior_type == "Laplace":
        prior = torch.distributions.laplace.Laplace(torch.zeros(ensemble.net.num_params).to(device),
                                torch.ones(ensemble.net.num_params).to(device))
    elif prior_type == "Cauchy":
        prior = torch.distributions.cauchy.Cauchy(torch.zeros(ensemble.net.num_params).to(device),
                                torch.ones(ensemble.net.num_params).to(device))
    elif prior_type == "StudentT":
        prior = torch.distributions.studentT.StudentT(3, torch.zeros(ensemble.net.num_params).to(device),
                                torch.ones(ensemble.net.num_params).to(device))
    elif prior_type == "Gaussian":
        prior = torch.distributions.normal.Normal(torch.zeros(ensemble.net.num_params).to(device),
                                torch.ones(ensemble.net.num_params).to(device))
    else:
        raise NotImplementedError("Priors Implemented: Gaussian, Cauchy, Laplace, StudentT")
        
    P = Unorm_post(ensemble, prior, data.num_train)

    #Specifying training updates
    method = create_method(P, optimizer, inference, temperature, epochs, data.num_train, lr_halving, device = device)

    #Training
    dataloader_train = data.get_train_dataloader(batch_size)
    for epoch in tqdm(range(epochs)):
        for (train_input, train_labels) in dataloader_train:
            optimizer.zero_grad()

            train_input_data = train_input.to(device)
            train_label_data = train_labels.to(device)

            driving, repulsive = method.step(W, train_input_data, train_label_data, epoch)


    test_data = data.test_inputs.to(device)
    test_labels = data.test_outputs.to(device)
            
    test_log_prob, test_pred = P.log_prob(W, test_data, test_labels, return_pred = True)
    averaged_pred = torch.mean(test_pred[0], 0)

    #Calculate Accuracy
    test_acc = (torch.argmax(averaged_pred, 1) == torch.argmax(test_labels, 1)).sum().item() / averaged_pred.shape[0] * 100

    #Calculate NLL
    averaged_logits = torch.mean(test_pred[1], 0)
    test_nll = F.cross_entropy(averaged_logits, test_labels.argmax(dim = 1))

    #Calculate ECE
    labels_ece = test_labels.argmax(dim = 1).detach().cpu().numpy()
    preds_ece = averaged_pred.detach().cpu().numpy()
    test_ece = ece(labels_ece, preds_ece)

    #Calculate RBS
    brier_score = ((averaged_pred-test_labels)**2).mean()
    test_rbs = torch.sqrt(brier_score).item()

    return ensemble, test_acc, test_nll.item(), test_ece, test_rbs