# Prior Choice and the Cold Posterior Effect in Practical Bayesian Neural Networks

This repo contains the code of the paper "Prior Choice and the Cold Posterior Effect in Practical Bayesian Neural Networks." We study how prior choice, approximate inference, and posterior temperature affect Bayesian neural network performance on MNIST, FashionMNIST, and CIFAR-10.

The experiments consider Gaussian, Laplace, Cauchy, and Student-t priors, with varying posterior temperatures from 1 to 10^(-7).

## Repository Structure

| Directory | Contents |
| --- | --- |
| `MFVI/` | Experiments with Mean-field variational inference (Bayes by Backprop) |
| `IVON/` | Experiments with Improved Variational Online Newton |
| `LLA/` | Experiments with Linearized Laplace approximation |
| `parameterParticles/` | Experiments with Parameter-space SVGD and WGD |
| `functionalParticles/` | Experiments with Function-space SVGD and WGD |
| `SGD/` | Deterministic SGD Baseline |
| `models/`, `utils/` | Architectures, data loading, metrics, and experiment settings |
