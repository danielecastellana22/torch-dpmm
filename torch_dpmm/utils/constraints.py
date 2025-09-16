import torch as th
from torch.linalg import eigvalsh

__all__ = ['BaseConstraint', 'AnyValue', 'GreaterThan', 'Positive', 'PositiveDefinite']


class BaseConstraint:
    """
    BaseConstraint serves as an abstract base class for defining constraints.

    This class provides a blueprint for implementing constraints, placing specific
    expectations to override its methods when subclassed. It includes mechanisms
    for checking constraints on values and generating appropriate messages when
    those constraints are violated.

    Methods:
        __call__(self, v): Abstract method to apply the constraint to a value.
        message(self, param_name, distr_name):
            Abstract method to generate a descriptive message indicating a
            constraint failure.
    """
    def __call__(self, v):
        raise NotImplementedError

    def message(self, param_name, distr_name):
        raise NotImplementedError


class AnyValue(BaseConstraint):
    """
    Represents a constraint that validates any value.

    The AnyValue class is a specific implementation of a constraint which allows every
    possible value to pass validation. It is designed to accept any input without
    evaluation or condition, effectively making it a universal validator.
    """
    def __call__(self, v):
        return True


class GreaterThan(BaseConstraint):
    """
    Represents a constraint that checks if values are strictly greater than a lower bound.

    This class is designed to impose constraints on parameter values, ensuring
    that all values are strictly greater than a specified lower bound.

    Attributes:
        lower_bound: The numerical lower bound that values must exceed.
    """
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, v):
        return th.all(v > self.lower_bound)

    def message(self, param_name, distr_name):
        return f'Params {param_name} of {distr_name} must be strictly greater than {self.lower_bound}.'


class Positive(GreaterThan):
    """
    A class used to enforce that numbers are positive.

    This class is a specialization of the `GreaterThan` class. It is initialized
    to validate numbers greater than 0.
    """
    def __init__(self):
        super().__init__(0)


class PositiveDefinite(BaseConstraint):
    """
    Represents a constraint that ensures a matrix is positive definite.

    This class is used to validate whether a matrix satisfies the condition
    of being positive definite.
    """
    def __call__(self, v):
        return th.all(eigvalsh(v) > 0)

    def message(self, param_name, distr_name):
        return f'Params {param_name} of {distr_name} must be positive definite.'
