import torch as th
from abc import ABC
from .base import ExponentialFamilyDistribution
from torch_dpmm.utils.constants import *
from torch_dpmm.utils.mat_utils import *
from torch_dpmm.utils.misc import *

__all__ = ['FullNIW', 'DiagonalNIW', 'SingleNIW', 'SphericalNormal']


class BaseNIW(ExponentialFamilyDistribution, ABC):
    r"""Represents the base class for the Normal-Inverse-Wishart (NIW) distribution as an
    exponential family distribution.

     .. :math:
        \mu, \Sigma \sim NIW(\mu_0, \lambda, \Phi, \nu)

    This class is a foundational implementation of the NIW distribution and follows
    the conventions and parameterization as described in related literature. The NIW
    distribution is commonly used as a prior distribution for a multivariate normal
    distribution with unknown mean and covariance. It is parameterized by four common
    parameters: `mu0`, `lambda`, `Phi`, and `nu`, and its natural parameters are
    denoted as `eta_1`, `eta_2`, `eta_3`, and `eta_4`. The class defines the structure
    of the distribution, relationships between parameters, and mathematical computations
    for its moments, sufficient statistics, and other properties. We follow https://arxiv.org/pdf/2405.16088v1.

    Methods
    -------
    _h_x(x: list[th.Tensor]) -> th.Tensor
        Computes the sufficient statistic function h(x) for the distribution.
    _A_eta(eta: list[th.Tensor]) -> th.Tensor
        Computes the log partition function for the given natural parameters.
    _T_x(x: list[th.Tensor], idx: int | None = None) -> list[th.Tensor] | th.Tensor
        Returns the sufficient statistics T(x) for specific indices or all indices.
    expected_T_x(eta: list[th.Tensor], idx: int | None = None) -> list[th.Tensor] | th.Tensor
        Computes the expected sufficient statistics T(x) under the distribution given
        the natural parameters.
    natural_to_common(eta: list[th.Tensor]) -> list[th.Tensor]
        Converts natural parameters to the common parameterization of the distribution.
    common_to_natural(theta: list[th.Tensor]) -> list[th.Tensor]
        Converts common parameters to the natural parameterization of the distribution.

    Attributes
    ----------
    _theta_names: list[str]
        Names of the common parameters for the distribution: ['mu0', 'lambda', 'Phi', 'nu'].
    _mat_ops_class: Type[BaseMatrixOperations] | None
        A reference to the matrix operations class used to handle matrix computations
        required by the distribution.

    Parameters
    ----------
    x : list[th.Tensor]
        Observations or variables associated with the distribution.
    eta : list[th.Tensor]
        Natural parameters of the NIW distribution.
    theta : list[th.Tensor]
        Common parameters of the NIW distribution.
    idx : int, optional
        Index to specify a specific sufficient statistic or moment.

    Returns
    -------
    th.Tensor | list[th.Tensor]
        The result of the computation, such as sufficient statistics, moments,
        or other properties, depending on the method invoked.

    Raises
    ------
    Any errors related to incorrect parameter types, shape mismatches, or other issues
    specific to the implementation or mathematical properties are not explicitly
    documented here but are expected in line with the implementation details.
    """
    _theta_names = ['mu0', 'lambda', 'Phi', 'nu']
    _mat_ops_class = None

    @classmethod
    def _h_x(cls, x: list[th.Tensor]) -> th.Tensor:
        cls._inner_h_x(*x)

    @classmethod
    def _inner_h_x(cls, mu, Sigma):
        D = mu.shape[-1]
        return th.pow(cls._mat_ops_class.det(Sigma), -0.5 * (D + 2)) * th.pow(PI, -0.5 * D)

    @classmethod
    def _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor:
        return cls._inner_A_eta(*cls.natural_to_common(eta))

    @classmethod
    def _inner_A_eta(cls, mu0, lam, Phi, nu):
        D = mu0.shape[-1]
        res = -0.5 * D * th.log(lam)
        res += -0.5 * nu * cls._mat_ops_class.log_det(Phi)
        res += 0.5 * D * nu * LOG_2
        res += th.special.multigammaln(0.5 * nu, D)
        return res

    @classmethod
    def _T_x(cls, x: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        return cls._inner_T_x(*x, idx)

    @classmethod
    def _inner_T_x(cls, mu, Sigma, idx):
        if idx == 0:
            return cls._mat_ops_class.inv_M_v(Sigma, mu)
        elif idx == 1:
            return -0.5 * cls._mat_ops_class.vT_inv_M_v(Sigma, mu)
        elif idx == 2:
            return -0.5 * cls._mat_ops_class.inv_M(Sigma)
        elif idx == 3:
            return -0.5 * cls._mat_ops_class.log_det(Sigma)
        else:
            return [cls._mat_ops_class.inv_M_v(Sigma, mu),
                    -0.5 * cls._mat_ops_class.vT_inv_M_v(Sigma, mu),
                    -0.5 * cls._mat_ops_class.inv_M(Sigma),
                    -0.5 * cls._mat_ops_class.log_det(Sigma)]

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        return cls._inner_expected_T_x(*cls.natural_to_common(eta), idx)

    @classmethod
    def _inner_expected_T_x(cls, mu0, lam, Phi, nu, idx):
        D = mu0.shape[-1]
        if idx == 0:
            return nu.unsqueeze(-1) * cls._mat_ops_class.inv_M_v(Phi, mu0)
        elif idx == 1:
            return -0.5 * D/lam - 0.5 * nu * cls._mat_ops_class.trace_M(cls._mat_ops_class.inv_M_v_vT(Phi, mu0))
        elif idx == 2:
            inv = cls._mat_ops_class.inv_M(Phi)
            return -0.5 * nu.view(nu.shape + (1,) * (inv.ndim - nu.ndim)) * inv
        elif idx == 3:
            return -0.5 * cls._mat_ops_class.log_det(Phi) + 0.5 * D * LOG_2 + 0.5 * multidigamma(0.5 * nu, D)
        else:
            inv = cls._mat_ops_class.inv_M(Phi)
            return [nu.unsqueeze(-1) * cls._mat_ops_class.inv_M_v(Phi, mu0),
                    -0.5 * D / lam - 0.5 * nu * cls._mat_ops_class.trace_M(cls._mat_ops_class.inv_M_v_vT(Phi, mu0)),
                    -0.5 * nu.view(nu.shape + (1,) * (inv.ndim - nu.ndim)) * inv,
                    -0.5 * cls._mat_ops_class.log_det(Phi) + 0.5 * D * LOG_2 + 0.5 * multidigamma(0.5 * nu, D)]

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        eta_1, eta_2, eta_3, eta_4 = eta
        lam = eta_2
        nu = eta_4
        mu0 = eta_1 / lam.unsqueeze(-1)
        Phi = eta_3 - lam.view(lam.shape + (1, )*(eta_3.ndim -1)) * cls._mat_ops_class.v_vT(mu0)
        return [mu0, lam, Phi, nu]

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        mu0, lam, Phi, nu = theta
        eta_1 = lam.unsqueeze(-1) * mu0
        eta_2 = lam
        eta_3 = Phi + lam.view(lam.shape + (1, )*(Phi.ndim -1)) * cls._mat_ops_class.v_vT(mu0)
        eta_4 = nu
        return [eta_1, eta_2, eta_3, eta_4]


class FullNIW(BaseNIW):
    """
    Class representing the FullNIW distribution.

    This class extends the BaseNIW class and provides functionality specific to
    the Full Normal-Inverse-Wishart (NIW) distribution. The FullNIW class is
    designed for cases where data follow a multivariate normal distribution
    with unknown means and covariances, governed by NIW priors. It includes
    specific constraints and shapes for its parameters.

    Attributes:
        _mat_ops_class: Specifies the operations class used for handling matrix
        operations specific to this NIW model.

        _theta_shape_list: List of parameter shapes used in the distribution, where:
            '[K, D]' indicates K distributions over D dimensions,
            '[K]' indicates a vector parameter,
            '[K, D, D]' indicates distributional parameters involving matrices,
            '[K]' functions as a scalar count parameter in the context.

        _theta_constraints_list: List of constraints for the parameters, where:
            'AnyValue()' indicates the parameter can take any value,
            'Positive()' requires the parameter to be positive,
            'PositiveDefinite()' requires the parameter to be a positive definite matrix,
            'GreaterThan(D+1)' requires the parameter to be strictly greater than D+1.
    """
    _mat_ops_class = FullMatOps
    _theta_shape_list = ['[K, D]', '[K]', '[K, D, D]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()', 'PositiveDefinite()', 'GreaterThan(D+1)']


class DiagonalNIW(BaseNIW):
    """
    Represents a Diagonal Normal-Inverse-Wishart (NIW) distribution.

    This class is a specific implementation of the Normal-Inverse-Wishart
    distribution where the covariance matrices are assumed to be diagonal. It
    inherits from the BaseNIW class and defines specific operations and
    constraints tailored for diagonal covariance handling.

    Attributes:
        _mat_ops_class (type): Specifies the operations class used for matrix
            manipulations in the distribution. Here, it is set to
            DiagonalMatOps, which is designed to handle diagonal matrices.
        _theta_shape_list (list): Holds the shapes of the parameter arrays used in
            the Diagonal NIW distribution. Each entry corresponds to one
            parameter's shape.
        _theta_constraints_list (list): Defines constraints for the parameters
            of the Diagonal NIW distribution. These constraints enforce valid
            values for each parameter to ensure mathematical correctness.
    """
    _mat_ops_class = DiagonalMatOps
    _theta_shape_list = ['[K, D]', '[K]', '[K, D]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()', 'Positive()', 'GreaterThan(D+1)']


class SingleNIW(DiagonalNIW):
    """
    Represents a Single Normal-Inverse-Wishart (NIW) distribution as a special case
    of a Diagonal Normal-Inverse-Wishart (DiagonalNIW).

    This class is specifically designed for handling NIW distributions with a
    single component. It extends the general DiagonalNIW and overrides some of
    its key methods to adapt functionality for the single-component case. It is
    useful in probabilistic modeling and Bayesian statistical applications that
    involve Normal-Inverse-Wishart priors. The class provides capabilities for
    conversions between natural and common parameters, computation of expected
    values, and transformation of data.

    Methods
    -------
    _h_x(cls, x: list[th.Tensor]) -> th.Tensor
        Computes a transformation on the provided data tensors specific
        to the SingleNIW model.

    _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor
        Calculates a value associated with the natural parameters of
        the SingleNIW model.

    _T_x(cls, x: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor
        Computes sufficient statistics for the given data under the
        SingleNIW setting.

    expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor
        Computes the expected sufficient statistics for the given natural
        parameters under the SingleNIW model.

    natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]
        Converts natural parameters into common parameters for the SingleNIW model.

    common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]
        Converts common parameters into natural parameters for the SingleNIW model.

    Raises
    ------
    This class does not describe raised errors in the documentation. Refer to the
    individual methods for potential exceptions.

    Attributes
    ----------
    The attributes and constraints for this class are inherited and pre-defined.
    Special attribute types such as `_theta_shape_list` and `_theta_constraints_list`
    are excluded from the documentation as they are internal.
    """
    _theta_shape_list = ['[K, D]', '[K]', '[K]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()', 'Positive()', 'GreaterThan(D+1)']

    @classmethod
    def _h_x(cls, x: list[th.Tensor]) -> th.Tensor:
        mu, s = x[0], x[1]
        D = mu.shape[-1]
        Sigma = s.view(-1, 1).expand(-1, D)
        return cls._inner_h_x(mu, Sigma)

    @classmethod
    def _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor:
        mu, lam, p, nu = cls.natural_to_common(eta)
        D = mu.shape[-1]
        Phi = p.view(-1, 1).expand(-1, D)
        return cls._inner_A_eta(mu, lam, Phi, nu)

    @classmethod
    def _T_x(cls, x: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        mu, s = x[0], x[1]
        D = mu.shape[-1]
        Sigma = s.view(-1, 1).expand(-1, D)
        ris = cls._inner_T_x(mu, Sigma, idx)
        if idx is None:
            ris[2] = ris[2].sum(-1)
        elif idx == 2:
            ris = ris.sum(-1)
        return ris

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        mu, lam, p, nu = cls.natural_to_common(eta)
        D = mu.shape[-1]
        Phi = p.view(-1, 1).expand(-1, D)
        ris = cls._inner_expected_T_x(mu, lam, Phi, nu, idx)
        if idx is None:
            ris[2] = ris[2].sum(-1)
        elif idx == 2:
            ris = ris.sum(-1)
        return ris

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        eta_1, eta_2, eta_3, eta_4 = eta
        lam = eta_2
        nu = eta_4
        mu0 = eta_1 / lam.unsqueeze(-1)
        D = mu0.shape[-1]
        p = eta_3 - lam * th.sum(mu0 * mu0, -1) / D
        return [mu0, lam, p, nu]

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        mu0, lam, p, nu = theta
        D = mu0.shape[-1]
        eta_1 = lam.unsqueeze(-1) * mu0
        eta_2 = lam
        eta_3 = p + lam * th.sum(mu0 * mu0, -1) / D
        eta_4 = nu
        return [eta_1, eta_2, eta_3, eta_4]


class SphericalNormal(ExponentialFamilyDistribution):
    """
    Represents a spherical normal distribution, which is a specific type of
    exponential family distribution.

    This class provides functionality for parameter conversion between natural
    and common forms, computation of sufficient statistics, log-partition
    functions, and expected sufficient statistics, specifically for spherical
    normal distributions where data has a fixed covariance.

    Attributes:
        theta_names: List of parameter names in the common parameterization
                     (mu, lam).
        theta_shape_list: List describing the expected shapes of the parameters
                          in the common parameterization.
        theta_constraints_list: List of constraints for each parameter to ensure
                                validity in the common parameterization.

    Methods:
        _h_x(x):
            Computes the base measure term for the distribution.

        _A_eta(eta):
            Computes the log-partition function for the series of natural
            parameters.

        _T_x(x, idx):
            Computes the sufficient statistics for a given sample or returns
            specific sufficient statistics if an index is provided.

        expected_T_x(eta, idx):
            Computes the expected sufficient statistics based on the given
            natural parameters, or returns specific expected statistics
            if an index is provided.

        natural_to_common(eta):
            Converts the natural parameters to the common parameterization.

        common_to_natural(theta):
            Converts the common parameters to the natural parameterization.
    """
    # x
    # x ~ UnitNormal(mu_0, lambda) = N(mu, 1/lam*I)
    # common params: mu_0, lambda
    # natural params: eta_1, eta_2

    _theta_names = ['mu', 'lam']
    _theta_shape_list = ['[K, D]', '[K]']
    _theta_constraints_list = ['AnyValue()', 'Positive()']

    @classmethod
    def _h_x(cls, x: list[th.Tensor]) -> th.Tensor:
        D = x[0].shape[-1]
        return th.pow(PI, -0.5 * D)

    @classmethod
    def _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor:
        mu, lam = cls.natural_to_common(eta)
        D = mu.shape[-1]
        return - 0.5 * D * th.log(lam) + 0.5 * lam * th.sum(mu**2, -1)

    @classmethod
    def _T_x(cls, x: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        x = x[0]
        if idx == 0:
            return x
        elif idx == 1:
            return th.sum(x**2, -1)
        else:
            return [x, -0.5 * th.sum(x**2, -1)]

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx=None) -> list[th.Tensor] | th.Tensor:
        mu, lam = cls.natural_to_common(eta)
        D = mu.shape[-1]
        if idx == 0:
            return mu
        elif idx == 1:
            return -0.5 * D/lam - 0.5 * th.sum(mu**2, -1)
        else:
            return [mu, -0.5 * D/lam - 0.5 * th.sum(mu**2, -1)]

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        eta_1, eta_2 = eta
        lam = eta_2
        mu0 = eta_1 / lam.unsqueeze(-1)
        return [mu0, lam]

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        mu0, lam = theta
        eta_1 = lam.unsqueeze(-1) * mu0
        eta_2 = lam
        return [eta_1, eta_2]


# TODO: implement cholesky parametrization of full covariance matrix to improve stability.


# TODO: maybe its reasonable to implement also LKJ prior
