import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.special import erf

SAVE_DPI = 300
SAVE_IMG_LOCATION = 'graphics'


def uniform_convergence(func1_arr, func2_arr):
    max_deviation = 0
    for i in range(len(func1_arr)):
        tmp_deviation = abs(func1_arr[i] - func2_arr[i])
        if tmp_deviation > max_deviation:
            max_deviation = tmp_deviation

    return max_deviation


def compare_deviations(deviations, labels, dots_amount_interpol, save_file=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, dpi=200)
    for i in range(len(deviations)):
        axes.semilogy(dots_amount_interpol, deviations[i], linestyle="--", marker='o', label=labels[i])

    axes.set_xlabel('Nodes (n)')
    axes.set_ylabel('Numerical error')
    axes.legend(loc='lower left')
    axes.grid()

    plt.show()

    if save_file:
        fig.savefig(SAVE_IMG_LOCATION + "/5_compare_err.png", dpi=SAVE_DPI)

    return


def erf_compare(erf_interpol, nodes, x, save_file=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, dpi=200)

    axes.semilogy(nodes, erf_interpol, linestyle="--", marker='o', label='Interpolated values')
    axes.semilogy(nodes, [erf(x) for i in range(len(nodes))], linestyle="-", label='Real value')

    axes.set_xlabel('Nodes (n)')
    axes.set_ylabel(r'Error function value for $x=2$')
    axes.legend(loc='upper right')
    axes.grid()

    plt.show()

    if save_file:
        fig.savefig(SAVE_IMG_LOCATION + "/6_erf.png", dpi=SAVE_DPI)

    return


class InterpolPlots:
    def __init__(self, x_plot, y_real, interpol_results, dots_amount, color_sheme="winter", graph_name='untitled',
                 dpi=200, save_file=False):
        self.x_plot = x_plot
        self.y_real = y_real
        self.interpol_results = interpol_results
        self.dots_amount_interpol = dots_amount
        self.color_map = plt.cm.get_cmap(color_sheme, len(dots_amount))
        self.graph_name = graph_name
        self.dpi = dpi
        self.save_file = save_file

    def show_interpol_results(self):
        fig, axes = plt.subplots(nrows=1, ncols=1, dpi=self.dpi)

        for i in range(len(self.dots_amount_interpol)):
            axes.plot(self.x_plot, self.interpol_results[i], color=self.color_map(i), linestyle="dotted")

        # See 'math expressions in matplotlib' for more info about how to write sub/superscripts
        axes.plot(self.x_plot, self.y_real, color="indigo", linestyle="solid", label=r'$f(x)$')
        axes.grid()

        # Writing custom mark in the legend, taking into account existing ones
        # Because this is the better way for legend family of Ln(x).
        lagrange_legend = plt.Line2D([], [], color=self.color_map(self.dots_amount_interpol[0]), linestyle="dotted",
                                     label=r'$L_n(x)$')

        # [0] at the end of expression means the first return value from the function
        handles = axes.get_legend_handles_labels()[0]
        handles.append(lagrange_legend)
        axes.legend(handles=handles, loc="upper left")

        # Creating mappable object for colorbar API
        sm = plt.cm.ScalarMappable(cmap=self.color_map, norm=plt.Normalize(vmin=4, vmax=20))
        color_bar = plt.colorbar(sm)
        color_bar.ax.set_ylabel('Nodes (n) amount')

        axes.set_xlabel(r'$x$')
        axes.set_ylabel(r'$y$')
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/3a_" + self.graph_name + ".png", dpi=SAVE_DPI)

        return

    def show_interpol_results_odd_even(self):
        interpol_results_even = self.interpol_results[0::2]
        interpol_results_odd = self.interpol_results[1::2]
        min_y = min(np.min(interpol_results_even), np.min(interpol_results_odd))
        max_y = max(np.max(interpol_results_even), np.max(interpol_results_odd))

        if (max_y - min_y) > (self.x_plot[-1] - self.x_plot[0]):
            nrows = 1
            ncols = 2
        else:
            nrows = 2
            ncols = 1

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 6), dpi=self.dpi)

        color_map_even = plt.cm.get_cmap('winter', len(self.dots_amount_interpol[0::2]))
        color_map_odd = plt.cm.get_cmap('winter', len(self.dots_amount_interpol[1::2]))

        for i in range(len(self.dots_amount_interpol[0::2])):
            axes[0].plot(self.x_plot, interpol_results_even[i], color=color_map_even(i), linestyle="dotted")

        for i in range(len(self.dots_amount_interpol[1::2])):
            axes[1].plot(self.x_plot, interpol_results_odd[i], color=color_map_odd(i), linestyle="dotted")

        belongs_sym = ['2k', '2k+1']
        for i in range(len(axes)):
            axes[i].set_ylim(min_y - 0.05, max_y + 0.05)
            axes[i].plot(self.x_plot, self.y_real, color="indigo", linestyle="solid", label=r'$f(x)$')
            axes[i].grid()
            lagrange_legend = plt.Line2D([], [], color=self.color_map(self.dots_amount_interpol[0]), linestyle="dotted",
                                         label=r'$L_n(x), n \in {}$'.format(belongs_sym[i]))
            handles = axes[i].get_legend_handles_labels()[0]
            handles.append(lagrange_legend)
            axes[i].legend(handles=handles, loc="upper left")
            axes[i].set_xlabel(r'$x$')
            axes[i].set_ylabel(r'$y$')

        # Creating mappable object for colorbar API
        sm1 = plt.cm.ScalarMappable(cmap=self.color_map, norm=plt.Normalize(vmin=4, vmax=20))
        sm2 = plt.cm.ScalarMappable(cmap=self.color_map, norm=plt.Normalize(vmin=5, vmax=19))
        sm = [sm1, sm2]
        ticks = [np.arange(4, 21, 2), np.arange(5, 20, 2)]

        for i in range(len(axes)):
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="3%", pad="3%")
            color_bar = plt.colorbar(sm[i], cax=cax)
            color_bar.set_ticks(ticks[i])
            color_bar.ax.set_ylabel('Nodes (n) amount')

        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/3a_" + self.graph_name + "-odd_even.png", dpi=SAVE_DPI)

        return

    def show_deviations(self, legend_loc='upper left'):
        deviations = []
        for i in range(len(self.dots_amount_interpol)):
            deviations.append(uniform_convergence(self.y_real, self.interpol_results[i]))

        fig, axes = plt.subplots(nrows=1, ncols=1, dpi=self.dpi)
        axes.scatter(self.dots_amount_interpol, deviations, color="blue", s=5, label="f(x)")
        axes.set_xlabel('Nodes (n)')
        axes.set_ylabel('Numerical error')
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/3b_" + self.graph_name + "-scatter.png", dpi=SAVE_DPI)

        fig, axes = plt.subplots(nrows=1, ncols=1, dpi=self.dpi)
        axes.plot(self.dots_amount_interpol[0::2], deviations[0::2], color="blue", linestyle="dashed", marker="o",
                  label=r'$n \in 2k$')
        axes.plot(self.dots_amount_interpol[1::2], deviations[1::2], color="green", linestyle="dashed", marker="o",
                  label=r'$n \in 2k+1$')
        axes.set_xlabel('Nodes (n)')
        axes.set_ylabel('Numerical error')
        axes.legend(loc=legend_loc)
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/3b_" + self.graph_name + "-plot.png", dpi=SAVE_DPI)

        return deviations

    def show_lagrange_analysis(self, deviations, legend_locs=('upper left', 'upper left')):
        fig, axes = plt.subplots(nrows=1, ncols=1, dpi=self.dpi)
        axes.scatter(self.dots_amount_interpol, deviations, color="blue", s=5, label="f(x)")
        axes.set_xlabel('Nodes (n)')
        axes.set_ylabel('Residual lagrange member estimate')
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/3c_" + self.graph_name + "-scatter.png", dpi=SAVE_DPI)

        fig, axes = plt.subplots(nrows=1, ncols=1, dpi=self.dpi)
        axes.plot(self.dots_amount_interpol[0::2], deviations[0::2], color="blue", linestyle="dashed", marker='o',
                  label=r'$n \in 2k$')
        axes.plot(self.dots_amount_interpol[1::2], deviations[1::2], color="green", linestyle="dashed", marker='o',
                  label=r'$n \in 2k+1$')
        axes.set_xlabel('Nodes (n)')
        axes.set_ylabel('Residual lagrange member estimate')
        axes.legend(loc=legend_locs[0])
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/3_c" + self.graph_name + "-plot.png", dpi=SAVE_DPI)

        comp_deviations = []
        for i in range(len(self.dots_amount_interpol)):
            comp_deviations.append(uniform_convergence(self.y_real, self.interpol_results[i]))

        fig, axes = plt.subplots(nrows=1, ncols=1, dpi=self.dpi)
        axes.semilogy(self.dots_amount_interpol, deviations, color="blue", linestyle="--", marker='o',
                      label="Analytical error")
        axes.semilogy(self.dots_amount_interpol, comp_deviations, color="green", linestyle="--", marker='o',
                      label="Numerical error")

        axes.set_xlabel('Nodes (n)')
        axes.set_ylabel('Error')
        axes.legend(loc=legend_locs[1])
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/3c_" + self.graph_name + "-comparison.png", dpi=SAVE_DPI)

        return
