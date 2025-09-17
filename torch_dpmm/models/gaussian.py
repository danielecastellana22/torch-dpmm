import torch as th
from .base import DPMM
from torch_dpmm.bayesian_distributions import FullNormalINIW, DiagonalNormalNIW, SingleNormalNIW, UnitNormalSpherical
from sklearn.cluster import kmeans_plusplus


__all__ = ['FullGaussianDPMM', 'DiagonalGaussianDPMM', 'UnitGaussianDPMM', 'IsotropicGaussianDPMM']


def _get_gaussian_init_vals(x, D, mask, v_c=None, v_n=None):
    """
    Computes initial values for a Gaussian mixture model's parameters, such as
    tau (cluster means), c, B (cluster variances), and n, based on the given
    inputs and initialization method.

    Parameters:
    x: Optional[torch.Tensor]
        Input tensor representing the data points. If provided, it will be used
        to calculate initialization values.
    D: int
        Dimensionality of the input data.
    mask: torch.Tensor
        Binary mask tensor indicating which clusters to initialize. Its sum
        represents the number of clusters to initialize.
    v_c: Optional[int]
        Value used for initializing c. Defaults to 1.
    v_n: Optional[int]
        Value used for initializing n. Defaults to D + 2.

    Raises:
    ValueError
        If the mask sum indicates that there are no clusters to initialize.

    Returns:
    tuple
        A tuple containing:
        - tau (torch.Tensor): Initialized cluster means with shape [K_to_init, D].
        - c (torch.Tensor): Initialized values for parameter c for each cluster.
        - B (torch.Tensor): Initialized cluster variance scalar.
        - n (torch.Tensor): Initialized values for parameter n for each cluster.
    """
    if v_c is None:
        v_c = 1

    if v_n is None:
        v_n = D+2

    K_to_init = th.sum(mask).item()
    if K_to_init == 0:
        raise ValueError('There are no clusters to initialise!')

    # compute initialisation for tau
    if x is None:
        tau = th.zeros([K_to_init, D], device=mask.device)
    else:
        x_np = x.detach().cpu().numpy()
        # initialisation makes the difference: we should cover the input space
        if x_np.shape[0] >= K_to_init:
            # there are enough sample to init all K_to_init clusters
            mean_np, _ = kmeans_plusplus(x_np, K_to_init)
            tau = th.tensor(mean_np, device=mask.device)
        else:
            # there are few samples
            to_init = x_np.shape[0]
            mean_np, _ = kmeans_plusplus(x_np, to_init)
            tau = th.zeros([K_to_init, D], device=mask.device)
            tau[:to_init] = th.tensor(mean_np, device=mask.device)

    # compute initialisation for B
    B = th.tensor(1.0, device=mask.device)
    if x is not None:
        B = th.var(x) * B

    # compute initialisation for c
    c = v_c * th.ones([K_to_init], device=mask.device)

    # compute initialisation for n
    n = v_n * th.ones([K_to_init], device=mask.device)

    return tau, c, B, n


def _to_common_params(D, mu_prior, mu_prior_strength, var_prior, var_prior_strength):
    mu0 = mu_prior
    lam = mu_prior_strength
    n = var_prior_strength + D + 1
    Phi = var_prior * n
    return mu0, lam, Phi, n


class FullGaussianDPMM(DPMM):
    """
    Represents a Gaussian Dirichlet Process Mixture Model (DPMM) with full covariance
    matrices.

    This class extends the base DPMM class and is specifically designed for use with Gaussian
    distributions possessing full covariance matrices. It allows the initialization of the
    model's hyperparameters and provides methods to handle parameters associated with
    the emission distributions. The model is commonly used for clustering and density
    estimation tasks when the data is expected to follow a Gaussian distribution.

    Attributes:
        K (int): The number of mixture components.
        D (int): The dimensionality of the data.
        alphaDP (float): The concentration parameter of the Dirichlet Process.
    """

    def __init__(self, K, D, alphaDP, mu_prior, mu_prior_strength, var_prior, var_prior_strength):
        """
        Initializes an instance of the FullGaussianDPMM class.

        This constructor sets up a Bayesian nonparametric model using a Dirichlet Process with a Gaussian
        likelihood and an inverse Wishart prior. It prepares the prior parameters based on the provided
        information and initializes the variational parameters for inference.

        Parameters:
            K (int): Number of clusters to use (truncation level for DP).
            D (int): Dimensionality of the data.
            alphaDP (float): Concentration parameter for the Dirichlet Process.
            mu_prior (Union[float, array-like]): Prior mean of the Gaussian distribution.
            mu_prior_strength (float): Strength of the prior mean relative to the data.
            var_prior (Union[float, array-like]): Prior variance (or covariance) for the Gaussian.
            var_prior_strength (float): Strength of the prior variance relative to the data.
        """
        mu0, lam, Phi, nu = _to_common_params(D, mu_prior, mu_prior_strength, var_prior, var_prior_strength)
        super(FullGaussianDPMM, self).__init__(K, D, alphaDP, FullNormalINIW, [mu0, lam, Phi, nu])
        self.init_var_params()

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask)
        B = th.diag_embed(B*th.ones_like(tau))
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class DiagonalGaussianDPMM(DPMM):
    """
    Represents a Dirichlet Process Mixture Model with diagonal Gaussian components.

    Detailed description of the class, its purpose, and usage.
    The DiagonalGaussianDPMM class is specifically designed to model data using
    diagonal Gaussian distributions as the components within a Dirichlet Process
    framework. The class supports initialization with specified hyperparameters
    and prior distributions for flexibility in various applications.

    Attributes:
        K (int): Number of components for the initialization of the model.
        D (int): Dimensionality of the data.
        alphaDP (float): Concentration parameter for the Dirichlet Process.
    """
    def __init__(self, K, D, alphaDP,  mu_prior, mu_prior_strength, var_prior, var_prior_strength):
        """
        Initializes a DiagonalGaussianDPMM class instance.

        The DiagonalGaussianDPMM class represents a Dirichlet Process Mixture Model
        with diagonal Gaussian components. The initialization incorporates prior
        distributions and hyperparameters for the model.

        Parameters:
            K (int): Number of components for the initialization of the model.
            D (int): Dimensionality of the data.
            alphaDP (float): Concentration parameter for the Dirichlet Process.
            mu_prior (float or array): Prior mean for the Gaussian distributions.
            mu_prior_strength (float): Strength of the prior on the mean.
            var_prior (float or array): Prior variance for the Gaussian distributions.
            var_prior_strength (float): Strength of the prior on the variance.
        """
        mu0, lam, Phi, nu = _to_common_params(D, mu_prior, mu_prior_strength, var_prior, var_prior_strength)
        super(DiagonalGaussianDPMM, self).__init__(K, D, alphaDP, DiagonalNormalNIW, [mu0, lam, Phi, nu])
        self.init_var_params()

    def _get_init_vals_emission_var_eta(self, x: th.Tensor = None, mask=None):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask)
        B = B*th.ones_like(tau)
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class IsotropicGaussianDPMM(DPMM):
    """
    Represents a Dirichlet Process Mixture Model (DPMM) with Isotropic Gaussian components.

    This class implements a Dirichlet Process Mixture Model where the data distribution in each cluster is modeled
    by a Gaussian distribution. The model uses a Normal-Inverse-Wishart prior for the emission distribution's sufficient
    statistics. It extends the base DPMM class and initializes its parameters and hyperparameters according to
    the provided prior specifications.

    Attributes:
        K (int): Number of initial clusters in the mixture model.
        D (int): Dimensionality of the data points.
        alphaDP (float): Concentration parameter for the Dirichlet Process.
        emission_distr_class (type): The type class describing the emission model distribution.
    """
    def __init__(self, K, D, alphaDP, mu_prior, mu_prior_strength, var_prior, var_prior_strength):
        """
        Initializes a Single-Gaussian Dirichlet Process Mixture Model (DPMM) with specified parameters.

        Parameters:
        K : int
            The number of components in the mixture model.
        D : int
            The dimensionality of the data.
        alphaDP : float
            The concentration parameter of the Dirichlet Process.
        mu_prior : list[float]
            Prior mean for the Gaussian components, given as a list matching the data dimensionality.
        mu_prior_strength : float
            Strength of the prior on the mean.
        var_prior : list[list[float]]
            Prior covariance matrix for the Gaussian components, represented as a 2D list.
        var_prior_strength : float
            Strength of the prior on the covariance.

        Raises:
        TypeError
            If the provided parameters do not match expected types.
        ValueError
            If invalid parameter values are supplied, such as negative strengths or mismatched dimensions
            for mu_prior or var_prior.

        """
        mu0, lam, Phi, nu = _to_common_params(D, mu_prior, mu_prior_strength, var_prior, var_prior_strength)
        super(IsotropicGaussianDPMM, self).__init__(K, D, alphaDP, SingleNormalNIW, [mu0, lam, Phi, nu])
        self.init_var_params()

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask)
        B = B * th.ones_like(c)
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class UnitGaussianDPMM(DPMM):
    """
    Represents a Dirichlet Process Mixture Model (DPMM) with unitary Gaussian components,
    i.e., Gaussian with identity covariance matrix.


    A specialized version of the Dirichlet Process Mixture Model (DPMM) where
    the emission distributions are unit Gaussian distributions. The unit Gaussian
    distributions are parameterized using natural parameters, and the model supports
    mean and variance initialization based on data. This class extends the base
    DPMM class by defining specific settings for Gaussian emissions and their
    corresponding natural parameter initialization.

    Attributes:
        None
    """
    def __init__(self, K, D, alphaDP,  mu_prior, mu_prior_strength):
        """
        Initialize a UnitGaussianDPMM instance.

        This constructor initializes a Dirichlet Process Mixture Model (DPMM) with
        Gaussian components assuming a spherical covariance structure. The parameters
        provided define the prior for the Gaussian components and control the overall
        DPMM structure.

        Parameters:
        K : int
            Number of initial clusters.
        D : int
            Dimensionality of the data.
        alphaDP : float
            Concentration parameter for the Dirichlet Process.
        mu_prior : float
            Mean of the Gaussian prior distribution.
        mu_prior_strength : float
            Strength (precision) of the Gaussian prior.

        Raises:
            No explicit errors are raised by this method directly.
        """
        mu0, lam = mu_prior, mu_prior_strength
        super(UnitGaussianDPMM, self).__init__(K, D, alphaDP, UnitNormalSpherical, [mu0, lam])
        self.init_var_params()

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, _, _ = _get_gaussian_init_vals(x, self.D, mask)
        return self.emission_distr_class.common_to_natural([tau, c])