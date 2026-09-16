"""Error models (likelihoods) used to fit and score candidate curves.

Every family exposes the same small interface so that candidates with different
error structures can be ranked on one scale: held-out negative log-likelihood.
"""
import numpy as np
from scipy.special import expit, gammaln, logit, xlogy

_EPS = 1e-12


class Family:
    name = ""
    label = ""
    extra_params = 0          # nuisance parameters estimated from residuals (e.g. the noise variance)
    link = "identity"

    def clip(self, mu):
        return mu

    def inverse_link(self, eta):
        return eta

    def working_response(self, y):
        """Rough link-scale version of y, used only to build starting values."""
        return y

    def fit_residuals(self, y, mu, w):
        """Residuals whose sum of squares is the quantity minimised when fitting."""
        raise NotImplementedError

    def scale(self, y, mu, w):
        return None

    def nll(self, y, mu, scale, w):
        """Per-observation negative log-likelihood, already multiplied by the weights."""
        raise NotImplementedError

    def mean(self, mu, scale):
        """Expected value of y given the fitted location."""
        return mu


class Gaussian(Family):
    name, label, extra_params = "gaussian", "加性误差（正态）", 1

    def fit_residuals(self, y, mu, w):
        return np.sqrt(w) * (y - mu)

    def scale(self, y, mu, w):
        return max(np.sum(w * (y - mu) ** 2) / np.sum(w), _EPS)

    def nll(self, y, mu, var, w):
        return w * (0.5 * np.log(2 * np.pi * var) + (y - mu) ** 2 / (2 * var))


class LogNormal(Family):
    """Multiplicative noise: log y = log f(x) + e. f(x) is the conditional median."""
    name, label, extra_params, link = "lognormal", "乘性误差（对数正态）", 1, "log"

    def clip(self, mu):
        return np.maximum(mu, 1e-300)

    def inverse_link(self, eta):
        return np.exp(np.clip(eta, -700, 700))

    def working_response(self, y):
        return np.log(y)

    def fit_residuals(self, y, mu, w):
        return np.sqrt(w) * (np.log(y) - np.log(self.clip(mu)))

    def scale(self, y, mu, w):
        r = np.log(y) - np.log(self.clip(mu))
        return max(np.sum(w * r**2) / np.sum(w), _EPS)

    def nll(self, y, mu, var, w):
        r = np.log(y) - np.log(self.clip(mu))
        return w * (np.log(y) + 0.5 * np.log(2 * np.pi * var) + r**2 / (2 * var))

    def mean(self, mu, var):
        return mu * np.exp(var / 2)


class Poisson(Family):
    name, label, link = "poisson", "计数（Poisson）", "log"

    def clip(self, mu):
        return np.clip(mu, 1e-10, None)

    def inverse_link(self, eta):
        return np.exp(np.clip(eta, -700, 700))

    def working_response(self, y):
        return np.log(y + 0.5)

    def fit_residuals(self, y, mu, w):
        mu = self.clip(mu)
        d = 2 * (xlogy(y, y / mu) - (y - mu))
        return np.sign(y - mu) * np.sqrt(np.maximum(w * d, 0))

    def nll(self, y, mu, _, w):
        mu = self.clip(mu)
        return w * (mu - xlogy(y, mu) + gammaln(y + 1))


class Binomial(Family):
    """Binary outcomes or observed proportions; weights are the number of trials."""
    name, label, link = "binomial", "概率/比例（二项）", "logit"

    def clip(self, mu):
        return np.clip(mu, 1e-9, 1 - 1e-9)

    def inverse_link(self, eta):
        return expit(eta)

    def working_response(self, y):
        return logit(np.clip(y, 0.05, 0.95))

    def fit_residuals(self, y, mu, w):
        mu = self.clip(mu)
        d = 2 * (xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu)))
        return np.sign(y - mu) * np.sqrt(np.maximum(w * d, 0))

    def nll(self, y, mu, _, w):
        mu = self.clip(mu)
        return -w * (xlogy(y, mu) + xlogy(1 - y, 1 - mu))


FAMILIES = {f.name: f for f in (Gaussian(), LogNormal(), Poisson(), Binomial())}


def get_family(name):
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(f"unknown family {name!r}; choose from {sorted(FAMILIES)}") from None
