import numpy as np
import scipy.stats as spstats
from decimal import Decimal

class CurveFitFuncs():
    def __init__(self):
        pass

    def remove_systematic_error(self, arr):
        return arr - arr[0]

    def residual(self, yarr_measured, yarr_predicted):
        return yarr_measured - yarr_predicted

    def sum_squared_ratio(self, numer, denom):
        return np.sum((numer ** 2) / (denom ** 2))

    def calc_dof(self, yarr_measured, params_in_model):
        dof = len(yarr_measured) - params_in_model
        return dof

    def calc_raw_chi_squared(self, yarr_measured, yarr_predicted, y_uncertainty):
        numer = self.residual(yarr_measured, yarr_predicted)
        denom = y_uncertainty
        return self.sum_squared_ratio(numer, denom)

    def calc_reduced_chi_squared(self, yarr_measured, yarr_predicted, y_uncertainty, params_in_model):
        numer = self.residual(yarr_measured, yarr_predicted)
        denom = y_uncertainty
        dof = len(yarr_measured) - params_in_model
        return self.sum_squared_ratio(numer, denom) / dof

    def calc_chi2_probability(self, raw_chi2, dof):
        chi2_prob = (1 - spstats.chi2.cdf(raw_chi2, dof))
        return chi2_prob


class CurveFitAnalysis():
    def __init__(self, xarr, yarr_measured, yarr_uncertainty, FittedFunc): # FittedFunc must have number_of_parameters attribute
        cff = CurveFitFuncs()
        yarr_predicted = FittedFunc(xarr)

        self.degrees_of_freedom = cff.calc_dof(yarr_measured, FittedFunc.number_of_parameters)
        self.raw_chi2 = cff.calc_raw_chi_squared(yarr_measured, yarr_predicted, yarr_uncertainty)
        self.reduced_chi2 = cff.calc_reduced_chi_squared(yarr_measured, yarr_predicted, yarr_uncertainty, FittedFunc.number_of_parameters)
        self.chi2_probability = cff.calc_chi2_probability(self.raw_chi2, self.degrees_of_freedom)



class Output():
    def __init__(self):
        pass

    def baseplot_errorbars(self, ax, x, y, yerr=None, xerr=None, **kwargs):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle='None', capsize=2, **kwargs)

    def baseplot_errorbars_with_markers(self, ax, x, y, yerr=None, xerr=None, **kwargs):
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, linestyle='None', capsize=2, marker='.', **kwargs)

    def get_dp(self, num): # returns number of decimal places
        decimal = Decimal(str(num))
        if decimal.as_tuple().exponent >= 0:
            dp = -int(np.log10(num))
        else:
            dp = -decimal.as_tuple().exponent
        return dp

    def to_sf(self, num, sf=3):
        # # result = '%.*g' % (sf, num)
        # # result = (f'{num:.{sf}g}')
        # result = '{:g}'.format(float('{:.{p}g}'.format(num, p=sf)))
        # # result = float(result)
        # # decimal = Decimal(str(result))
        # # print(decimal.as_tuple().exponent)
        # # if decimal.as_tuple().exponent >= 0:
        # if len(result.split('.')) == 1 or all(result.split('.')) == '0':
        #     result = int(result)
        # else:
        #     result = float(result)
        # return result
        result = np.format_float_positional(num, precision=sf, fractional=False, trim='-')
        # if '.' in result:
        #     result = float(result)
        # else:
        #     result = int(result)
        return result

    def print_with_uncertainty(self, num, uncertainty):
        rounded_uncertainty = self.to_sf(uncertainty, sf=1)
        uncertainty_dp = self.get_dp(rounded_uncertainty)
        rounded_num = round(num, uncertainty_dp)
        rounded_num_dp = self.get_dp(rounded_num)

        if uncertainty_dp <= 0:
            rounded_num = int(rounded_num)

        rounded_uncertainty = str(rounded_uncertainty)
        rounded_num = str(rounded_num)

        if uncertainty_dp > 0 and rounded_num_dp <= 0:
            rounded_num += '.'
            for i in range(uncertainty_dp - rounded_num_dp):
                rounded_num += '0'

        return '$' + rounded_num + ' \pm ' + rounded_uncertainty + '$'


# Output = Output()
# # num = 3453478.2981732
# # uncert = 6.64
# # decimal = Decimal(str(uncert))
# # print(decimal.as_tuple().exponent, Decimal(str(Output.to_sf(6.64, 1))).as_tuple().exponent, '??')
# # print(Output.to_sf(6.64, 1))
# # print(Output.print_with_uncertainty(num, uncert))
# # print(int(round(6.64, 0)))
# uncert = 6.64
# print(Output.to_sf(uncert, 1))
# print(Output.get_dp(uncert))
# print(Output.get_dp(Output.to_sf(uncert, 1)))