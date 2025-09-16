from typing import Any, Collection
import torch as th
from torch_dpmm.utils.constraints import *

__all__ = ['ExponentialFamilyDistribution']


class ExponentialFamilyDistribution:
    """
    Represents an abstraction for an Exponential Family Distribution.

    The class provides a structured interface to work effectively with
    Exponential Family Distributions. It focuses on defining and validating
    key distribution parameters, and it outlines methods for implementing
    important mathematical properties like base measure (h(x)), sufficient statistics
    (T(x)), log-partition function (A(eta)), and transformations between natural
    and common parameters.

    This class is designed to be subclassed to create specific Exponential Family
    Distributions. Subclasses should implement the methods necessary for their specific
    use case to compute the properties and transformations defined in this class.
    """
    # eta: natural parameters of the distribution
    # theta: common parameters of the distribution
    # x: a sample of the distribution

    _theta_names = None
    _theta_shape_list = None
    _theta_constraints_list = None

    @staticmethod
    def __validate_params(name: str, value: Any,
                          expected_shape: Collection[int], constraint: BaseConstraint):
        expected_shape = tuple(expected_shape)

        if not isinstance(value, th.Tensor):
            value = th.tensor(value).float()

        if value.ndim == 0:
            if isinstance(constraint, PositiveDefinite):
                value = value.view((1,) * len(expected_shape[:-1])).expand(expected_shape[:-1])
                value = th.diag_embed(value)
            else:
                value = value.view((1,) * len(expected_shape)).expand(expected_shape)

        # check the shape
        if value.ndim > len(expected_shape):
            raise ValueError(f'{name} has too many dimensions: we got {value.shape} but we expected {expected_shape}!')
        elif value.ndim == len(expected_shape)-1:
            value = value.unsqueeze(0).expand(expected_shape[:1] + tuple([-1]*(len(expected_shape)-1)))
        elif value.ndim < len(expected_shape)-1:
            raise ValueError(f'{name} has too few dimensions! We broadcast only along the first dimension:'
                             f' we got {value.shape} but we expected {expected_shape}!')

        assert value.ndim == len(expected_shape)
        if value.shape != expected_shape:
            raise ValueError(f'{name} has the wrong shape: we got {value.shape} but we expected {expected_shape}!')

        # check the constraint
        if not constraint(value):
            raise ValueError(constraint.message(name, 'Gaussian-DPMM'))

        return value

    @classmethod
    def validate_common_params(cls, K: int, D: int, theta: list[th.Tensor]) -> list[th.Tensor]:
        out = []
        for i in range(len(theta)):
            out.append(cls.__validate_params(cls._theta_names[i], theta[i],
                                             eval(cls._theta_shape_list[i]), eval(cls._theta_constraints_list[i])))
        return out


    @classmethod
    def _h_x(cls, x: list[th.Tensor]) -> th.Tensor:
        """ The base measure h(x).

        Args:
            x (list[th.Tensor]):  the samples of the distribution. It is a list of tensors, eahc with size BS x ?.

        Returns:
            th.Tensor: the value of h(x).
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def _A_eta(cls, eta: list[th.Tensor]) -> th.Tensor:
        """ The log-partition function A(eta).

        Args:
            eta (list[th.Tensor]): the natural parameters. It is a list of tensors of size K x ?.

        Returns:
            th.Tensor: the value of A(eta).
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def _T_x(cls, x: list[th.Tensor], idx: int = None) -> list[th.Tensor] | th.Tensor:
        """ The sufficient statistics T(x). If idx is not None, we return only the statistic at in position idx.

        Args:
            x (list[th.Tensor]): the samples of the distribution. It is a list of tensors, each with size BS x ?.
            idx (int): the idx of the parameter we want to retrieve

        Returns:
            list[th.Tensor]: the list of the sufficient statistics T(x). It is a list of tensors if idx is None.
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def expected_T_x(cls, eta: list[th.Tensor], idx: int = None) -> list[th.Tensor] | th.Tensor:
        """ The expected value of the sufficient statistics T(x), i.e., E_{P(x|eta)}[T(x)].
        The expectation is equal to the derivative of A(eta) w.r.t. eta.
        If idx is not None, we return only the expectation of the statistic at in position idx.

        Args:
            eta (list[th.Tensor]): the natural parameters. It is a list of tensors of size K x ?.
            idx (int): the idx of the parameter we want to retrieve

        Returns:
             list[th.Tensor]: the expectation of each sufficient statistics. It is a list of tensors if idx is None.
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def natural_to_common(cls, eta: list[th.Tensor]) -> list[th.Tensor]:
        """ Transform the natural parameters to the common ones.

        Args:
            eta (list[th.Tensor]): the natural parameters. It is a list of tensors of size K x ?.

        Returns:
             list[th.Tensor]: the common parameters. It is a list of tensors of size K x ? if idx is None
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def common_to_natural(cls, theta: list[th.Tensor]) -> list[th.Tensor]:
        """ Transform the common parameters to the natural ones.

        Args:
            theta (list[th.Tensor]): the common parameters. It is a list of tensors of size K x ?.

        Returns:
             list[th.Tensor]: the natural parameters. It is a list of tensors of size K x ? if idx is None
        """
        raise NotImplementedError('Should be implemented in subclasses!')

    @classmethod
    def kl_div(cls, p_eta: list[th.Tensor], q_eta: list[th.Tensor]):
        """ Compute the KL divergence between p and q.

        Args:
            p_eta (list[th.Tensor]): the common parameters of p.
            q_eta (list[th.Tensor]): the common parameters of q.

        Returns:
            th.Tensor: the value of the kl divergence.
        """
        div = 0
        exp_T = cls.expected_T_x(p_eta)
        for i in range(len(p_eta)):
            term = (p_eta[i] - q_eta[i]) * exp_T[i]
            if p_eta[i].ndim > 1:
                term = th.sum(term, list(range(1, p_eta[i].ndim)))
            div += term
        div += -cls._A_eta(p_eta) + cls._A_eta(q_eta)
        return div
