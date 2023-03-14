import numpy as np
import fitted_models


class Gaussian:  # height is usually 1; not probability density
    def __init__(self):
        self.number_of_parameters = 4
        self.CorrespondingFittedFunction = fitted_models.Gaussian

    def __call__(self, x, base, scale, mu, sigma):
        result = base + scale * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return result


class GaussianZeroCenter:  # height is usually 1; not probability density
    def __init__(self):
        self.number_of_parameters = 3
        self.CorrespondingFittedFunction = fitted_models.GaussianZeroCenter

    def __call__(self, x, base, scale, sigma):
        result = base + scale * np.exp(-0.5 * ((x) / sigma) ** 2)
        return result


class Linear:
    def __init__(self):
        self.number_of_parameters = 2
        self.CorrespondingFittedFunction = fitted_models.Linear

    def __call__(self, x, m, c):
        result = m * x + c
        return result

class Proportional:
    def __init__(self):
        self.number_of_parameters = 1
        self.CorrespondingFittedFunction = fitted_models.Proportional

    def __call__(self, x, m):
        result = m * x
        return result

class Constant:
    def __init__(self):
        self.number_of_parameters = 1
        self.CorrespondingFittedFunction = fitted_models.Constant

    def __call__(self, x, c):
        result = c
        return result


class DecayingSinusoid:
    def __init__(self):
        self.number_of_parameters = 5
        self.CorrespondingFittedFunction = fitted_models.DecayingSinusoid
        # self.parameter_bounds = ([-np.inf, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, 2*np.pi, np.inf])

    def __call__(self, t, base, amplitude, period, phase, exponential_factor):
        result = base + np.exp(-t * exponential_factor) * amplitude * np.sin((2 * np.pi / period) * t + phase)
        return result