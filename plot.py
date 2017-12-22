"""

 plot.py  (author: Anson Wong / git: ankonzoid)


 Copyright 2017 Anson Wong

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

"""
import os, csv, sys
import numpy as np
import matplotlib.pyplot as plt

def main():

    env_name = "survive_1"

    D = read_extract_csv(os.path.join("output", env_name, "log.csv"))
    red = (195/255, 0/255, 0/255)
    blue = (13/255, 28/255, 164/255)
    if 0:
        save_plot("episode",
                  ["rew_0", "rew_1"],
                  [blue, red],
                  ["agent-0", "agent-1"],
                  D, env_name, ["episode", "reward"])
        save_plot("episode",
                  ["loss_0", "loss_1"],
                  [blue, red],
                  ["agent-0", "agent-1"],
                  D, env_name, ["episode", "loss"])
    else:
        save_plot("episode",
                  ["rew_0"],
                  [blue],
                  ["agent-0"],
                  D, env_name, ["episode", "reward"])
        save_plot("episode",
                  ["loss_0"],
                  [blue],
                  ["agent-0"],
                  D, env_name, ["episode", "loss"])

def read_extract_csv(input_file):

    def _read_csv(csv_filename):
        FID = open(csv_filename, 'rt')
        try:
            reader = csv.reader(FID)
            data_mat = []
            column_headers = []
            for i, row in enumerate(reader):
                if i == 0:
                    column_headers = row
                else:
                    data_mat.append(row)
            column_headers = np.array(column_headers)
            data_mat = np.array(data_mat)
            n_rows, n_columns = len(data_mat), len(column_headers)
        finally:
            FID.close()
        return (column_headers, data_mat, n_rows, n_columns)

    D = {}
    headers, M, n_rows, n_columns = _read_csv(input_file)  # M = numpy array of strings
    for i, header in enumerate(headers):
        D[header] = M[:, i]
    return D

def save_plot(x_name, y_names, colors, labels,
              D, env_name, axis_labels, n_bins=50):

    def moving_average(a, n_bins):
        ret = np.cumsum(a, dtype=float)
        ret[n_bins:] = ret[n_bins:] - ret[:-n_bins]
        return ret[n_bins - 1:] / n_bins

    # Moving average bin size
    if len(D[x_name]) <= n_bins:
        sys.exit("We have n_bin_ma = {} and not enough data points!".format(n_bins))

    # Plot
    print("Saving {} vs {} for '{}'...".format(y_names, x_name, env_name))
    for i in range(len(y_names)):
        y_name, color, label = y_names[i], colors[i],labels[i]
        (x_plot, y_plot) = (np.array(D[x_name], dtype=np.float), np.array(D[y_name], dtype=np.float))
        y_moving_avg = moving_average(y_plot, n_bins)
        plt.plot(x_plot, y_plot, marker=".", markersize=1, color=color, linestyle='None', alpha=0.6)
        plt.plot(x_plot[(n_bins - 1):], y_moving_avg, "-", color=color, label=label)

    # Settings
    plt.title(env_name)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Save
    output_file = os.path.join("output", env_name, "{}_{}.pdf".format(axis_labels[1], axis_labels[0]))
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()

# Driver
if __name__ == "__main__":
    main()