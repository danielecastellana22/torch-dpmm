import torch as th
from .base import DPMM
from torch_dpmm.bayesian_distributions import FullNormalINIW, DiagonalNormalNIW, SingleNormalNIW, UnitNormalSpherical
from sklearn.cluster import kmeans_plusplus
import warnings


__all__ = ['FullGaussianDPMM', 'DiagonalGaussianDPMM', 'UnitGaussianDPMM', 'SingleGaussianDPMM']


def _get_gaussian_init_vals(x, D, mask, v_c=None, v_n=None):
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

    def __init__(self, K, D, alphaDP, mu_prior, mu_prior_strength, var_prior, var_prior_strength):
        mu0, lam, Phi, nu = _to_common_params(D, mu_prior, mu_prior_strength, var_prior, var_prior_strength)
        super(FullGaussianDPMM, self).__init__(K, D, alphaDP, FullNormalINIW, [mu0, lam, Phi, nu])
        self.init_var_params()

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask)
        B = th.diag_embed(B*th.ones_like(tau))
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class DiagonalGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP,  mu_prior, mu_prior_strength, var_prior, var_prior_strength):
        mu0, lam, Phi, nu = _to_common_params(D, mu_prior, mu_prior_strength, var_prior, var_prior_strength)
        super(DiagonalGaussianDPMM, self).__init__(K, D, alphaDP, DiagonalNormalNIW, [mu0, lam, Phi, nu])
        self.init_var_params()

    def _get_init_vals_emission_var_eta(self, x: th.Tensor = None, mask=None):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask)
        B = B*th.ones_like(tau)
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class SingleGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP, mu_prior, mu_prior_strength, var_prior, var_prior_strength):
        mu0, lam, Phi, nu = _to_common_params(D, mu_prior, mu_prior_strength, var_prior, var_prior_strength)
        super(SingleGaussianDPMM, self).__init__(K, D, alphaDP, SingleNormalNIW, [mu0, lam, Phi, nu])
        self.init_var_params()

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, B, n = _get_gaussian_init_vals(x, self.D, mask)
        B = B * th.ones_like(c)
        return self.emission_distr_class.common_to_natural([tau, c, B, n])


class UnitGaussianDPMM(DPMM):

    def __init__(self, K, D, alphaDP,  mu_prior, mu_prior_strength):
        mu0, lam = mu_prior, mu_prior_strength
        super(UnitGaussianDPMM, self).__init__(K, D, alphaDP, UnitNormalSpherical, [mu0, lam])
        self.init_var_params()

    def _get_init_vals_emission_var_eta(self, x: th.Tensor | None, mask):
        tau, c, _, _ = _get_gaussian_init_vals(x, self.D, mask)
        return self.emission_distr_class.common_to_natural([tau, c])