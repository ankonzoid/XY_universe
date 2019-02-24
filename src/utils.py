"""

 utils.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import os
import matplotlib.pyplot as plt

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        if not os.path.exists(dir):
            raise Exception("Could not create directory '{}'.format(dir")
    return dir

def save_plot(columns, df, color, n_bins):
    x_name, y_name = columns[0], columns[1]
    def windowed_average(a, n_bins):
        ret = np.cumsum(a, dtype=float)
        ret[n_bins:] = ret[n_bins:] - ret[:-n_bins]
        return ret[n_bins - 1:] / n_bins
    assert len(df[x_name].values) > n_bins
    outFile = os.path.join("output", "{}_{}.pdf".format(y_name, x_name))
    print("Plotting {} vs {} into '{}'...".format(y_name, x_name, outFile))
    x_plot, y_plot = df[x_name].values, df[y_name].values
    y_moving_avg = windowed_average(y_plot, n_bins)
    plt.plot(x_plot, y_plot, marker=".", markersize=1, color=color, linestyle='None', alpha=0.6)
    plt.plot(x_plot[(n_bins - 1):], y_moving_avg, "-", color=color)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(outFile, bbox_inches='tight')
    plt.close()