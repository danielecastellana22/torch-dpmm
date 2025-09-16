# Pytorch implementation of Dirichlet Process Mixture Model (DPMM)

We integrate the Variational Inference framework with the autograd functionality of pytorch. To this end, we have extended pytorch autograd by defining a Function for the computation of DPMM:
- in the **forward pass**, we compute the cluster assignment and the elbo given the data and the model parameters;
- in the **backward pass**, we compute the natural gradient of the model parameters.

This allows to train DPMM without changing the training loop (which becomes the same of a neural model):
```
from torch_dpmm.models import GaussianDPMM
import torch.optim as optim

th_DPMM = FullGaussianDPMM(K=K, D=D, alphaDP=10, tau0=0, c0=1, n0=3*D, B0=1)
optimiser = optim.SGD(params=m.parameters(), lr=0.1)

for i in range(num_epochs):
        optimiser.zero_grad()
        pi, elbo_loss = th_DPMM(x)
        elbo_loss.backward()
        optimiser.step()
```

There are four types of Gaussian DPMMs that differs in terms of parametrisation of the covariance matrix:
1. `UnitGaussianDPMM` defines an identity covariance matrix (no parameters);
2. `SingleGaussianDPMM` defines a covariance matrix in the form $sI$, where s is a scalar parameter;
3. `DiagonalGaussianDPMM` defines a diagonal covariance matrix $D$, where all the element on the diagonal can be adjusted during the training 
4. `FullGaussianDPMM` defines a fully trainable covariance matrix. 


We provide a complete example [here](examples/first_example.ipynb).
We recommend to download the notebook to view the animations.

### Installation
You can install *torch-dpmm* by running the command:
```
pip install git+https://github.com/danielecastellana22/torch-dpmm.git@main 
```

To execute the examples, you need to install also other packages by executing the command:
```
pip install tqdm matplotlib jupyter-lab
```

## Implementation Details
The implementation is fully based on the natural parametrisation of the [exponential family distribution](https://en.wikipedia.org/wiki/Exponential_family). This allows to compute the natural gradient of the variational parameters in a straightforward way. Check this [paper](https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf) for more details on the Stochastic Variational Inference (SVI) framework and the computation of the natural gradient of the variational parameters.

There are three main abstract classes:
1. `ExponentialFamilyDistribution`
2. `BayesianDistribution`
3. `DPMM`

### ExponentialFamilyDistribution
`ExponentialFamilyDistribution` represents a distribution of the exponential family by defining the base measure, the log-partition function, the sufficient statistics. It also provides utility functions such as the mapping between the common and the natural parametrisation and the computation of the KL divergence.

### BayesianDistribution
`BayesianDistribution` represents a Bayesian distributions by using conjugate priors. Let $P(x | \phi)$ the distribution of interest, this class define a distribution $Q(\phi | \eta)$ over the parameter $\phi$; $Q$ is the conjugate prior of $P$ and $\eta$ are the variational parameters. We represent $Q$ by using the class `ExponentialFamilyDistribution`.

### DPMM
`DPMM` represents a DPMM which is fully determined by specifying a Dirichlet Process (DP) prior over the mixture weights and an emission distribution. The DP prior is approximated through the truncated Stick-Breaking distribution. Both the emission and mixture weights distribution are defined as object of type `BayesianDistribution`.

The computation of the Elbo and the assignments are executed by defining a new pytorhc autograd function. This allows to compute the gradient of the variational parameters by calling `elbo.backward()`.

## To Do
1. Add Cholesky parametrisation of the full Guassian distribution to improve numerical stability.
2. Improve the initialisation of the variational parameter.
