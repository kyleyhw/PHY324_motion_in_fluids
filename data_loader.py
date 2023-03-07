import numpy as np


from fitting_and_analysis import CurveFitFuncs
cff = CurveFitFuncs()

class DataLoader():
    def __init__(self, filename):
        directory = 'data/%s.txt' % (filename)
        self.full_data = np.loadtxt(directory, dtype=float, skiprows=2).T

        raw_positions = self.full_data[1][self.full_data[1] != 0]
        raw_times = self.full_data[0][self.full_data[1] != 0]
        raw_times = raw_times[raw_positions < 250]
        raw_positions = raw_positions[raw_positions < 250]

        raw_positions = raw_positions / 1000 # mm to m

        raw_times = cff.remove_systematic_error(raw_times)

        self.y = raw_positions
        self.y_error = np.zeros_like(self.y) + 0.5

        self.x = raw_times
        self.x_error = np.zeros_like(self.x) + 0.005

