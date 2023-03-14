import numpy as np

import fit_models
from fitting_and_analysis import Output

Output = Output()


class Gaussian:  # height is usually 1; not probability density
    def __init__(self, popt, parameter_errors, info_sigfigs=2):
        self.popt = popt
        (self.base, self.scale, self.mu, self.sigma) = popt

        self.function = fit_models.Gaussian()
        self.number_of_parameters = self.function.number_of_parameters

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result


class GaussianZeroCenter:  # height is usually 1; not probability density
    def __init__(self, popt, parameter_errors, info_sigfigs=2):
        self.popt = popt
        (self.base, self.scale, self.sigma) = popt

        self.function = fit_models.GaussianZeroCenter()
        self.number_of_parameters = self.function.number_of_parameters

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result


class Linear:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.m, self.c) = popt
        (self.m_error, self.c_error) = parameter_errors

        self.function = fit_models.Linear()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('m', 'c')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class Proportional:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.m) = popt[0]
        (self.m_error) = parameter_errors[0]

        self.function = fit_models.Proportional()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('m')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)


    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class Constant:
    def __init__(self, popt, parameter_errors, units_for_parameters):
        self.popt = popt
        (self.c) = popt[0]
        (self.c_error) = parameter_errors[0]

        self.function = fit_models.Constant()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('c')
        self.units_for_parameters = units_for_parameters

        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, x):
        result = self.function(x, *self.popt)
        return result

class DecayingSinusoid:
    def __init__(self, popt, parameter_errors, units_for_parameters=None):
        self.popt = popt
        (self.B, self.A, self.T, self.tau, self.phi) = popt
        (self.B_error, self.A_error, self.T_error, self.tau_error, self.phi_error) = parameter_errors

        self.function = fit_models.DecayingSinusoid()
        self.number_of_parameters = self.function.number_of_parameters
        self.parameter_names = ('B', 'A', 'T', r'$\phi$', r'$\frac{1}{\tau}$')
        if units_for_parameters == None:
            self.units_for_parameters = tuple(np.zeros(shape=self.number_of_parameters, dtype=str))
        else:
            self.units_for_parameters = units_for_parameters

        self.parameter_info_list = [str(self.parameter_names[i]) + ' = (' + Output.print_with_uncertainty(self.popt[i],
                                                                                                          parameter_errors[
                                                                                                              i]) + ') ' +
                                    self.units_for_parameters[i] for i in range(self.number_of_parameters)]
        self.parameter_info = '\n'.join(self.parameter_info_list)

    def __call__(self, t):
        result = self.function(t, *self.popt)
        return result