import matplotlib.pyplot as plt
import fitting_and_analysis
Output = fitting_and_analysis.Output()
import fitting

import data_loader
import fit_models

fluids = ['water', 'glycerol']
bead_numbers = range(1, 6)


def plot_raw_graph(ax, fluid, bead_number, trial):
    filename = '%s/%s' %(fluid, fluid + 'bead' + str(bead_number) + 't' + str(trial))
    data = data_loader.DataLoader(filename)
    Output.baseplot_errorbars(ax=ax, x=data.x, y=data.y, xerr=data.x_error, yerr=data.y_error)

def plot_raw_for_bead(fluid, bead_number, show=False, save=False):
    number_of_trials = 5
    trials = range(1, number_of_trials+1)
    fig, axs = plt.subplots(number_of_trials, 1, figsize=(16,9), sharex=True)
    for i in trials:
        index = i-1
        plot_raw_graph(ax=axs[index], fluid=fluid, bead_number=bead_number, trial=i)

    fig.suptitle('Position vs time for bead size %s in %s' %(str(bead_number), fluid))
    for ax in axs:
        ax.set_ylabel('position / mm')
    axs[-1].set_xlabel('time / s')

    if save:
        plt.savefig('plots/raw_plots/%s.png' %(fluid + '_bead' + str(bead_number) + '_raw_plots'))
    if show:
        plt.show()

def plot_raw_all_beads(show=False, save=False):
    for fluid in fluids:
        for bead_number in bead_numbers:
            plot_raw_for_bead(fluid=fluid, bead_number=bead_number, show=show, save=save)

def plot_fit_graph(ax, fluid, bead_number, trial, model):
    filename = '%s/%s' % (fluid, fluid + 'bead' + str(bead_number) + 't' + str(trial))
    data = data_loader.DataLoader(filename)

    units_for_parameters = ('m / s')

    fit = fitting.Fitting(model=model, x=data.x, x_error=data.x_error, y_measured=data.y, y_error=data.y_error,
                              units_for_parameters=units_for_parameters, p0=(100))

    fit.scatter_plot_data_and_fit(ax=ax)

def plot_fit_for_bead(model, fluid, bead_number, show=False, save=False):
    number_of_trials = 5
    trials = range(1, number_of_trials+1)
    fig, axs = plt.subplots(number_of_trials, 1, figsize=(16,9), sharex=True)
    for i in trials:
        index = i-1
        plot_fit_graph(ax=axs[index], fluid=fluid, bead_number=bead_number, trial=i, model=model)

    fig.suptitle('Position vs time for bead size %s in %s' %(str(bead_number), fluid))
    for ax in axs:
        ax.set_ylabel('position / m')
    axs[-1].set_xlabel('time / s')

    if save:
        plt.savefig('plots/fit_plots/%s.png' %(fluid + '_bead' + str(bead_number) + '_raw_plots'))
    if show:
        plt.show()

def plot_fit_all_beads(model, show=False, save=False):
    for fluid in fluids:
        for bead_number in bead_numbers:
            plot_fit_for_bead(model=model, fluid=fluid, bead_number=bead_number, show=show, save=save)


# plot_raw_all_beads(show=False, save=True)

model = fit_models.Proportional()
plot_fit_all_beads(model=model, show=False, save=False)
