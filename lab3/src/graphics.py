import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

SAVE_DPI = 300
SAVE_IMG_LOCATION = '../graphics'
FIGSIZE = (8, 6)


class LorentzPlots:
    def __init__(self, dpi=200, save_file=False):
        self.dpi = dpi
        self.save_file = save_file

    def show_results(self, data, stationary=None, labels=None, name=""):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=self.dpi, subplot_kw={'projection': '3d'})
        for i in range(len(data)):
            x_values = [coords[0] for coords in data[i]]
            y_values = [coords[1] for coords in data[i]]
            z_values = [coords[2] for coords in data[i]]

            if labels is None:
                axes.plot(x_values, y_values, z_values, linewidth=0.5)
            else:
                axes.plot(x_values, y_values, z_values, linewidth=0.5, label=labels[i])
        if stationary:
            x_stat = [coords[0] for coords in stationary]
            y_stat = [coords[1] for coords in stationary]
            z_stat = [coords[2] for coords in stationary]
            axes.scatter(x_stat, y_stat, z_stat, color="navy", alpha=1, s=8, label="Stationary points")
            axes.legend(loc='best')

        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')
        if labels:
            axes.legend(loc='best')
        # axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/2_results_" + name + ".png", dpi=SAVE_DPI)

    def show_projection(self, data, projection="xy", labels=None, name=""):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=self.dpi)
        for i in range(len(data)):
            x_values = [coords[0] for coords in data[i]]
            y_values = [coords[1] for coords in data[i]]
            z_values = [coords[2] for coords in data[i]]

            if projection == "xy":
                line, = axes.plot(x_values, y_values, linewidth=0.5)
            elif projection == "xz":
                line, = axes.plot(x_values, z_values, linewidth=0.5)
            elif projection == "yz":
                line, = axes.plot(y_values, z_values, linewidth=0.5)
            else:
                print("invalid projection: {}".format(projection))
                return
            if labels:
                line.set_label(labels[i])

        axes.set_xlabel(projection[0])
        axes.set_ylabel(projection[1])
        if labels:
            axes.legend(loc='best')
        # axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/4_projection_" + name + ".png", dpi=SAVE_DPI)

    def show_times(self, data, h_steps, labels, name=""):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=self.dpi)
        for i in range(len(data)):
            axes.plot(h_steps, data[i], linestyle="--", marker='o', label=labels[i])

        axes.set_xlabel("h")
        axes.set_ylabel("Calculation time (in seconds)")
        if labels:
            axes.legend(loc='best')
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/5_times_" + name + ".png", dpi=SAVE_DPI)