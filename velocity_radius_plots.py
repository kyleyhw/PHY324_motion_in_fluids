import matplotlib.pyplot as plt
from matplotlib import rc
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
rc('font', **font)
import numpy as np

import fitting_and_analysis
Output = fitting_and_analysis.Output()
import fitting
import data_loader
import fit_models

fluids = ['water', 'glycerol']
bead_numbers = range(1, 6)

velocities = np.loadtxt('velocities.txt', delimiter=',', dtype=str)
print(velocities)