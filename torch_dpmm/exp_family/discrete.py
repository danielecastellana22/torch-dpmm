import torch as th
from .base import ExponentialFamilyDistribution

__all__ = ['Beta']


class Beta(ExponentialFamilyDistribution):
    """
    Represents the Beta distribution in the exponential family form.

    This class implements the Beta distribution with the parameterization suitable for
    exponential family distributions. It defines properties and methods required for
    transformations between natural and common parameters, computation of sufficient
    statistics, and other mathematical operations.

    Attributes:
        _theta_names: List of strings representing the names of the common parameters ['alpha', 'beta'].
        _theta_shape_list: List specifying the shapes of each of the common parameters.
        _theta_constraints_list: List that defines constraints on the values of the common parameters.

    Methods:
        _h_x: Computes the base measure 'h(x)' of the distribution.
        _A_eta: Computes the log normalizing constant 'A(eta)' given natural parameters.
        _T_x: Computes the sufficient statistics 'T(x)' of the distribution.
        expected_T_x: Computes the expected sufficient statistics given the natural parameters.
        natural_to_common: Converts the natural parameters to the common parameterization.
        common_to_natural: Converts the common parameters to the natural parameterization.
    """
    # x ~ Beta(alpha, beta)
    # common params: [alpha, beta]
    # natural params: [eta_1, eta_2]

    _theta_names = ['aplha', 'beta']
    _theta_shape_list = ['[K]', '[K]']
    _theta_constraints_list = ['Positive()', 'Positive()']

    @classmethod
    def _h_x(cls, x):
        return 1/(x[0] * (1-x[0]))

    @classmethod
    def _A_eta(cls, eta):
        return th.lgamma(eta[0]) + th.lgamma(eta[1]) - th.lgamma(eta[0] + eta[1])

    @classmethod
    def _T_x(cls, x, idx=None):
        if idx == 0:
            return th.log(x[0])
        elif idx == 1:
            return th.log(1 - x[0])
        else:
            return [th.log(x[0]), th.log(1-x[0])]

    @classmethod
    def expected_T_x(cls, eta, idx=None):
        aux = th.digamma(eta[0] + eta[1])
        if idx == 0:
            return th.digamma(eta[0]) - aux
        elif idx == 1:
            return th.digamma(eta[1]) - aux
        else:
            return [th.digamma(eta[0]) - aux, th.digamma(eta[1]) - aux]

    @classmethod
    def natural_to_common(cls, eta):
        return eta

    @classmethod
    def common_to_natural(cls, theta):
        return theta
