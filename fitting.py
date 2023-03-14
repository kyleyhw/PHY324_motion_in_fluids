import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rc
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
rc('font', **font)
from matplotlib.offsetbox import AnchoredText

from fitting_and_analysis import CurveFitFuncs
from fitting_and_analysis import CurveFitAnalysis
from fitting_and_analysis import Output
Output = Output()

class Fitting():
    def __init__(self, model, x, y_measured, y_error, x_error=None, p0=None, units_for_parameters=None): # requires hard coding for now
        self.model = model
        self.x = x
        self.x_error = x_error
        self.y_measured = y_measured
        self.y_error = y_error

        self.popt, self.pcov = curve_fit(self.model, self.x, self.y_measured, sigma=y_error, absolute_sigma=True, p0=p0, maxfev=100000)
        self.parameter_errors = np.sqrt(np.diag(self.pcov))

        self.fitted_function = self.model.CorrespondingFittedFunction(popt=self.popt, parameter_errors=self.parameter_errors, units_for_parameters=units_for_parameters)

        self.optimal_parameters = {self.fitted_function.parameter_names[i] : self.popt[i] for i in range(len(self.fitted_function.parameter_names))}
        self.error_in_parameters = {self.fitted_function.parameter_names[i]: self.popt[i] for i in range(len(self.fitted_function.parameter_names))} # terrible variable naming

        self.y_predicted = self.fitted_function(self.x)

        self.cfa = CurveFitAnalysis(self.x, self.y_measured, self.y_error, self.fitted_function)

    def scatter_plot_data_and_fit(self, ax, plot_fit=True, **kwargs):

        Output.baseplot_errorbars(ax=ax, x=self.x, y=self.y_measured, yerr=self.y_error, xerr=self.x_error, label='data')

        if plot_fit:
            x_for_plotting_fit = np.linspace(self.x[0], self.x[-1], 10000)
            y_for_plotting_fit = np.zeros_like(x_for_plotting_fit) + self.fitted_function(x_for_plotting_fit)

            ax.plot(x_for_plotting_fit, y_for_plotting_fit, label='fit', linewidth=3)

            info_sigfigs = 3
            info_fontsize = 14

            info_on_ax = self.fitted_function.parameter_info + \
                         '\n$\chi^2$ / DOF = ' + str(Output.to_sf(self.cfa.raw_chi2, sf=info_sigfigs)) + ' / ' + str(self.cfa.degrees_of_freedom) + ' = ' + str(Output.to_sf(self.cfa.reduced_chi2, sf=info_sigfigs)) + \
                         '\n$\chi^2$ prob = ' + str(np.format_float_positional(self.cfa.chi2_probability, precision=info_sigfigs, trim='k', min_digits=info_sigfigs))


            ax_text = AnchoredText(info_on_ax, loc='upper left', frameon=False, prop=dict(fontsize=info_fontsize))
            ax.add_artist(ax_text)

        ax.legend(loc='upper right')

        self.data_plot_xlim = ax.get_xlim()

    def plot_residuals(self, ax):
        cff = CurveFitFuncs()

        residuals = cff.residual(self.y_measured, self.y_predicted)
        error_in_residuals = np.zeros_like(residuals) + 0.001 # np.sqrt((self.parameter_errors[0] * residuals) ** 2 + (self.parameter_errors[1]) ** 2 + (self.y_error) ** 2)  # ErrorPropagation.add(ErrorPropagation.add(self.parameter_errors[0] * residuals, self.parameter_errors[1]), self.y_error)

        Output.baseplot_errorbars_with_markers(ax=ax, x=self.x, y=residuals, yerr=error_in_residuals, xerr=None,
                                               label='residuals')

        ax.set_xlim(*self.data_plot_xlim)

        ax.legend(loc='upper right')