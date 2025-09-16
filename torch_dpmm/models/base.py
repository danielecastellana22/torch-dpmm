from typing import Type
import torch as th
import torch.nn as nn
from torch_dpmm.utils.misc import log_normalise
from torch_dpmm.bayesian_distributions import BayesianDistribution, CategoricalSBP
from torch.autograd.function import once_differentiable
from torch.autograd import Function
from torch_dpmm import _DEBUG_MODE


class DPMMFunction(Function):
    """
    Implements the necessary functions for the Variational Inference (VI) update of
    a Dirichlet Process Mixture Model (DPMM).

    This class overrides PyTorch's autograd capabilities to perform the classical variational parameters update.
    During the forward pass, it computes the ELBO of the data, and all the sufficient statistics.
    Conversely, in the backward-pass, it applies the variational parameters update formula by using the sufficient statistics.

    The class works with different emission distribution.

    Attributes:
        None
    """
    @staticmethod
    def forward(ctx, data, emission_distr_class, prior_eta, *var_eta):
        mix_weights_prior_eta, emission_prior_eta = prior_eta[:2], prior_eta[2:]
        mix_weights_var_eta, emission_var_eta = var_eta[:2], var_eta[2:]
        pi_contribution = CategoricalSBP.expected_log_params(mix_weights_var_eta)[0]
        data_contribution = emission_distr_class.expected_data_loglikelihood(data, emission_var_eta)

        log_unnorm_r = pi_contribution.unsqueeze(0) + data_contribution
        log_r, log_Z = log_normalise(log_unnorm_r)
        r = th.exp(log_r)

        # compute the elbo
        elbo = ((r * data_contribution).sum()
                - CategoricalSBP.kl_div(mix_weights_var_eta, mix_weights_prior_eta).sum()
                - emission_distr_class.kl_div(emission_var_eta, emission_prior_eta).sum()
                # KL(q(z) || p(z)) where z is the cluster assignment
                + (r.sum(0) * pi_contribution).sum() - (r * log_r).sum())

        ctx.save_for_backward(r, data, *var_eta)
        ctx.emission_distr_class = emission_distr_class
        ctx.prior_eta = prior_eta

        return r, -elbo / th.numel(data), log_Z

    @staticmethod
    @once_differentiable
    def backward(ctx, pi_grad, elbo_grad, log_Z_grad):
        emission_distr_class = ctx.emission_distr_class
        prior_eta = ctx.prior_eta
        r, data, *var_eta = ctx.saved_tensors

        var_eta_suff_stasts = CategoricalSBP.compute_posterior_suff_stats(r) + \
                              emission_distr_class.compute_posterior_suff_stats(r, data)

        var_eta_updates = [prior_eta[i] + var_eta_suff_stasts[i] for i in range(len(prior_eta))]

        if _DEBUG_MODE:
            K, D = r.shape[-1], data.shape[-1]
            sbp_updates = var_eta_updates[:2]
            emiss_updates = var_eta_updates[2:]
            CategoricalSBP.validate_common_params(K, D, CategoricalSBP.natural_to_common(sbp_updates))
            emission_distr_class.validate_common_params(K, D, emission_distr_class.natural_to_common(emiss_updates))

        # The natural gradient is the difference between the current value and the new one
        # We also consider elbo_grad to mimic the backpropagation. It should be always 1.
        var_eta_grads = [(var_eta[i] - var_eta_updates[i]) * elbo_grad for i in range(len(var_eta))]

        # there is no gradient for the data x
        return (None, None, None) + tuple(var_eta_grads)


class DPMM(nn.Module):
    """
    Implementation of a Dirichlet Process Mixture Model (DPMM) in a PyTorch framework.

    This class leverages Bayesian distributions and stick-breaking processes for
    defining mixture models. It manages priors for both mixture weights and emissions
    by allowing EM-like updates using variational inference. Meant to be subclassed
    for specific emission distributions.

    Attributes:
        K (int): Number of mixture components.
        D (int): Size of output vector.
        emission_distr_class (Type[BayesianDistribution]): Class of the emission distribution.
        mix_weights_var_eta (list[nn.Parameter]): Variational parameters for mixture weights.
        emission_var_eta (list[nn.Parameter]): Variational parameters for emission distribution.
    """
    def __init__(self, K, D, alphaDP,
                 emission_distr_class: Type[BayesianDistribution], emission_prior_theta: list[th.Tensor]):
        """
        Initializes an instance of the class with specified parameters for a Bayesian mixture model.

        This initializer sets up the number of components (K), the size of the output vector (D), and
        configures the specified Bayesian emission distribution. It also initializes the prior and
        variational parameters for both the mixture weights and the emission distributions, ensuring
        compatibility and proper parameterization based on the given prior natural parameters.

        Attributes:
        K: int
            The number of mixture components in the model.
        D: int
            The dimensionality of the output vector.
        emission_distr_class: Type[BayesianDistribution]
            The class representing the Bayesian emission distribution.
        emission_prior_theta: list[th.Tensor]
            The prior natural parameters for the emission distribution.

        Parameters:
        K: int
            Number of mixture components.
        D: int
            Dimensionality of the output vector.
        alphaDP: float
            Concentration parameter for the Dirichlet Process.
        emission_distr_class: Type[BayesianDistribution]
            Class representing the emission distribution, which needs to come with required methods
            for parameter validation and transformation.
        emission_prior_theta: list[th.Tensor]
            List of tensors representing the prior common parameters for the emission distributions.

        Raises:
        TypeError: If emission_distr_class is not a subclass of BayesianDistribution.
        ValueError: If any parameter does not meet the expected requirements by the Bayesian structures
            or validation methods.
        """
        super().__init__()

        self.K = K  # number of mixture components
        self.D = D  # size of output vector
        self.emission_distr_class = emission_distr_class

        # store the prior nat params of the mixture weights and create the variational parameters
        mix_weights_prior_theta = CategoricalSBP.validate_common_params(K, D, [1, alphaDP])
        self.mix_weights_var_eta = []
        for i, p in enumerate(CategoricalSBP.common_to_natural(mix_weights_prior_theta)):
            b_name = f'mix_prior_eta_{i}'
            p_name = f'mix_var_eta_{i}'
            self.register_buffer(b_name, p.contiguous())
            self.register_parameter(p_name, nn.Parameter(th.empty_like(p)))
            self.mix_weights_var_eta.append(self.get_parameter(p_name))

        emission_prior_theta = emission_distr_class.validate_common_params(K, D, emission_prior_theta)
        self.emission_var_eta = []
        for i, p in enumerate(emission_distr_class.common_to_natural(emission_prior_theta)):
            b_name = f'emission_prior_eta_{i}'
            p_name = f'emission_var_eta_{i}'
            self.register_buffer(b_name, p.contiguous())
            self.register_parameter(p_name, nn.Parameter(th.empty_like(p)))
            self.emission_var_eta.append(self.get_parameter(p_name))
            
    @property
    def mix_weights_prior_eta(self):
        """
        Fetches the prior distribution values for mixture weights.

        This property retrieves a list of prior distribution parameters (eta) for
        the mixture weights. The values are obtained from buffers named
        `mix_prior_eta_<index>`, where `<index>` corresponds to each component
        index in the mixture model. These buffers are expected to store the
        previously defined or updated parameters for the mixture weights prior.

        Returns
        -------
        list
            A list of prior distribution parameters (eta) for the mixture weights.
        """
        return [self.get_buffer(f'mix_prior_eta_{i}') for i in range(len(self.mix_weights_var_eta))]

    @property
    def emission_prior_eta(self):
        return [self.get_buffer(f'emission_prior_eta_{i}') for i in range(len(self.emission_var_eta))]

    def forward(self, x):
        """
        Computes the forward pass of the DPMMFunction by applying it to the input
        and relevant distribution parameters.

        Parameters:
        x: Input tensor for which the DPMM model is applied.
        self.emission_distr_class: The distribution class defining the emission
          distribution.
        self.mix_weights_prior_eta: Prior dirichlet parameters for the mixture
          weights.
        self.emission_prior_eta: Prior distribution parameters for the emission
          distribution, concatenated to the mixture weights prior.
        self.mix_weights_var_eta: Variational dirichlet parameters for the mixture
          weights.
        self.emission_var_eta: Variational distribution parameters for the emission
          distribution, concatenated to the mixture weights variational parameters.

        Returns:
        Tensor containing the result of applying the DPMMFunction to the input
        tensor and model parameters.
        """
        return DPMMFunction.apply(x, self.emission_distr_class,
                                  self.mix_weights_prior_eta + self.emission_prior_eta,  # concatenate the eta lists
                                  *(self.mix_weights_var_eta + self.emission_var_eta))  # concatenate the eta lists

    def init_var_params(self, x=None, mask=None, mix_init_theta=None, emission_init_theta=None):
        """
        Initializes variational parameters with optional initial values.

        This method sets the initial values for the variational parameters of the
        mixture model and emission distributions. It allows optional initial
        values for the mixture weights and emission distributions. If no
        initial values are provided, default values or computed values are
        assigned to the variational parameters.

        Parameters:
            x: Optional[torch.Tensor]
                Data tensor used for initializing emission variational parameters,
                if emission_init_theta is not provided.
            mask: Optional[torch.BoolTensor]
                Boolean tensor indicating which components are valid or active.
                Defaults to all components being valid.
            mix_init_theta: Optional[Any]
                Common parameter for initializing the mixture weights in natural
                parameter space.
            emission_init_theta: Optional[Any]
                Common parameter for initializing the emission distributions in natural
                parameter space.

        Raises:
            ValueError: If input parameters fail validation checks in the respective
                        distribution classes.
        """
        if mask is None:
            mask = th.ones(self.K, dtype=th.bool, device=self.mix_weights_var_eta[0].device)

        K = th.sum(mask)
        mix_init_eta = None
        if mix_init_theta is not None:
            mix_init_theta = CategoricalSBP.validate_common_params(K, self.D, mix_init_theta)
            mix_init_eta = CategoricalSBP.common_to_natural(mix_init_theta)

        for i, p in enumerate(self.mix_weights_var_eta):
            if mix_init_eta is not None:
                p.data[mask] = mix_init_eta[i]
            else:
                p.data[mask] = 1

        if emission_init_theta is not None:
            emission_init_theta = self.emission_distr_class.validate_common_params(K, self.D, emission_init_theta)
            emission_init_eta = self.emission_distr_class.common_to_natural(emission_init_theta)
        else:
            emission_init_eta = self._get_init_vals_emission_var_eta(x, mask)

        for i, p in enumerate(self.emission_var_eta):
            p.data[mask] = emission_init_eta[i]

    def _get_init_vals_emission_var_eta(self, x, mask):
        """
        Initializes emission variable eta for emission distributions.

        This method calculates initialization values for the emission variable eta
        using provided data and mask inputs.

        Raises
        ------
        NotImplementedError
            Indicates that this method must be implemented in subclasses.
        """
        raise NotImplementedError('This should be implmented in the sublcasses!')

    @th.no_grad()
    def get_var_params(self):
        """
        Returns the variational parameters of the model in their common (non-natural) parameterization
        by converting the stored natural parameters. This method utilizes the natural-to-common parameter
        transformation functions defined within the relevant classes.

        Returns:
            Generator: A generator that yields the detached variational parameters in their common
            parameterization.
        """
        params = CategoricalSBP.natural_to_common(self.mix_weights_var_eta) + \
                 self.emission_distr_class.natural_to_common(self.emission_var_eta)

        return (p.detach() for p in params)

    @th.no_grad()
    def get_num_active_components(self):
        """
        Computes the number of active components in a categorical mixture model.

        The function calculates the expected parameters of the mixture weights using
        the Stick-breaking Process (SBP) and counts the number of components that
        have an expected probability greater than 0.01. It is intended to provide a
        measure of the significant or active components in the mixture.

        Returns
        -------
        int
            The number of active components in the mixture model.
        """
        r = CategoricalSBP.expected_params(self.mix_weights_var_eta)[0]
        return th.sum(r > 0.01).item()

    @th.no_grad()
    def get_expected_params(self):
        """
        Fetches the expected parameters for the distributions.

        This function retrieves the expected parameters for the mixing weights
        and emission distributions using the specified internal variables.
        The resulting values are detached from the computation graph.

        Returns:
            tuple: A tuple containing two elements:
                - A tensor representing the detached expected parameters for
                  the mixing weights.
                - A list of tensors representing the detached expected parameters
                  for the emission distributions.
        """
        r = CategoricalSBP.expected_params(self.mix_weights_var_eta)[0].detach()
        expected_emission_params = [v.detach() for v in self.emission_distr_class.expected_params(self.emission_var_eta)]

        return r, expected_emission_params

