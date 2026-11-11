# Prior Choice and the Cold Posterior Effect in Practical Bayesian Neural Networks

This repo contains the code of the paper "Prior Choice and the Cold Posterior Effect in Practical Bayesian Neural Networks." We study how prior choice, approximate inference, and posterior temperature affect Bayesian neural network performance on MNIST, FashionMNIST, and CIFAR-10.

The experiments consider Gaussian, Laplace, Cauchy, and Student-t priors, with varying posterior temperatures from 1 to 10^(-7).

## Repository Structure

The repository is split by approximate inference method with the experiments for each approximate inference method located its own separate directory with WGD and SVGD found under `parameterParticles/` and the functional variations found under `functionalParticles/`. `models` and `utils` contain architectures, data loading, metrics, and experiment settings across the various experiments.
