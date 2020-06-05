import matplotlib.pyplot as plt
import numpy as np

SAVE_DPI = 300
SAVE_IMG_LOCATION = '../graphics'
FIGSIZE = (8, 6)


class MatricesPlots:
    def __init__(self, matrices_data, dpi=200, save_file=False):
        self.dpi = dpi
        self.matrices_data = matrices_data
        self.save_file = save_file

    def show_hist(self, data, columns_number=50, label=None, x_label="", name=""):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=self.dpi)

        if label:
            axes.hist(data, bins=columns_number, rwidth=0.7, log=True, label=label)
            axes.legend(loc='best')
        else:
            axes.hist(data, bins=columns_number, rwidth=0.7, log=True)

        axes.set_xlabel(x_label)
        axes.set_ylabel("Distribution")

        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/" + name + ".png", dpi=SAVE_DPI)
