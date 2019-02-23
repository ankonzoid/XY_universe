"""

 plot.py  (author: Anson Wong / git: ankonzoid)


 Copyright 2019 Anson Wong

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
import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv(os.path.join("output", "log.csv"))
    print("Plotting...")
    save_plot(["episode", "reward"], df, color=(13/255, 28/255, 164/255), n_bins=50)
    save_plot(["episode", "loss"], df, color=(195/255, 0/255, 0/255), n_bins=50)

def save_plot(columns, df, color, n_bins):
    x_name, y_name = columns[0], columns[1]
    def windowed_average(a, n_bins):
        ret = np.cumsum(a, dtype=float)
        ret[n_bins:] = ret[n_bins:] - ret[:-n_bins]
        return ret[n_bins - 1:] / n_bins
    assert len(df[x_name].values) > n_bins
    print("Saving {} vs {}...".format(x_name, y_name))
    x_plot, y_plot = df[x_name].values, df[y_name].values
    y_moving_avg = windowed_average(y_plot, n_bins)
    plt.plot(x_plot, y_plot, marker=".", markersize=1, color=color, linestyle='None', alpha=0.6)
    plt.plot(x_plot[(n_bins - 1):], y_moving_avg, "-", color=color)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    output_file = os.path.join("output", "{}_{}.pdf".format(y_name, x_name))
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()

# Driver
if __name__ == "__main__":
    main()