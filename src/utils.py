"""

 utils.py  (author: Anson Wong / git: ankonzoid)

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as mc
from matplotlib.patches import Wedge

def mkdirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def windowed_avg(x, y, n_bins):
    yavg = np.cumsum(y, dtype=float)
    yavg[n_bins:] = yavg[n_bins:] - yavg[:-n_bins]
    return x[(n_bins-1):], yavg[n_bins-1:] / n_bins

def save_plot(xname, yname, df, color, n_bins):
    assert len(df[xname].values) > n_bins
    outFile = os.path.join("output", "{}_{}.png".format(yname, xname))
    print("Plotting {} vs {} into '{}'...".format(yname, xname, outFile))
    x, y = df[xname].values, df[yname].values
    xavg, yavg = windowed_avg(x, y, n_bins)
    plt.plot(x, y, marker=".", markersize=1, color=color, linestyle='None', alpha=0.6)
    plt.plot(xavg, yavg, "-", color=color)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outFile, bbox_inches='tight')
    plt.close()

def save_animation(agent, env, filename):
    print("    -> saving animation = {}".format(filename), flush=True)
    j_x, j_y, j_type = env.state_dict["x"], env.state_dict["y"], env.state_dict["type"]
    j_xy = [env.state_dict["x"], env.state_dict["y"]]
    orange, light_orange = _color(255, 153, 51), _color(244, 164, 96)
    brown, light_brown = _color(198, 74, 12), _color(222, 184, 135)
    dark_red, red = _color(220, 20, 60), _color(255, 153, 153)
    dark_blue, blue = _color(65, 105, 225), _color(153, 204, 255)
    dark_grey, grey = _color(89, 89, 89), _color(169, 169, 169)
    almost_black = _color(30, 30, 30)
    type_color_dict = {-2: brown, -1: orange, 0: dark_grey, 1: dark_red, 2: dark_blue}
    sensor_type_color_dict = {-2: light_brown, -1: light_orange, 0: dark_grey, 1: red, 2: blue}
    if type_color_dict.keys() != sensor_type_color_dict.keys():
        raise Exception("non-matching keys for our particle and sensed color dictionaries!")
    # Build animation
    fig, ax = plt.subplots(tight_layout=True)
    x_min, x_max = -env.box_half_length, env.box_half_length
    y_min, y_max = -env.box_half_length, env.box_half_length
    ax.margins(0.1, tight=True) # axes settings
    ax.set_aspect(abs(x_max - x_min) / abs(y_max - y_min)) # set aspect ratio to be square
    # Start
    def animate(i):
        ax.clear()  # clear up objects that were previously added
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # Draw particles (all types)
        coll_list = []
        for type in type_color_dict:
            facecolor = type_color_dict[type]
            zorder = 3 if (type < 0) else 2
            radius_draw = env.radius_agent if (type < 0) else env.radius_particles
            idxs = np.where(np.array(env.state_history[i][:, j_type], dtype=np.int) == type)[0]
            centers = env.state_history[i][idxs][:, j_xy]
            radii = radius_draw * np.ones(len(idxs), dtype=np.float)
            patches = [plt.Circle(center, radius) for (center, radius) in zip(centers, radii)]
            coll = mc.PatchCollection(patches, facecolors=facecolor, edgecolors=almost_black, linewidth=1, zorder=zorder)
            ax.add_collection(coll)
            coll_list.append(coll)
        # Make observation and draw sectors
        ob, sector_observation, sectors = agent.sensors.sense(env, force_sense_custom_state=True, custom_state=env.state_history[i])
        sensor_patches, color_patches = [], []
        for j, (sector_ob, sector) in enumerate(zip(sector_observation, sectors)):
            # Type of wedge (based on which one has lowest sector_obs)
            all_same_obs = all([x == 1.0 for x in sector_ob])  # check if all == 1
            idx_type_min_obs = sector_ob.index(min(sector_ob))
            type_min_obs = env.types_detect[idx_type_min_obs]
            # Set theta (in degrees)
            if j == 0:
                theta_1, theta_2 = 0 * (180 / np.pi), sectors[j] * (180 / np.pi)
            else:
                theta_1, theta_2 = sectors[j-1] * (180 / np.pi), sectors[j] * (180 / np.pi)
            # Draw wedge
            sensor_ob = (sector_ob[idx_type_min_obs] * (agent.sensors.sector_length - env.radius_agent)) + env.radius_agent
            obs_sensor_length = sensor_ob if sensor_ob >= 0 else 0.0
            wedge = Wedge((env.state_history[i][0][j_x], env.state_history[i][0][j_y]), obs_sensor_length, theta_1, theta_2, alpha=0.5)
            # Append wedge shape
            sensor_patches.append(wedge)
            # Append wedge color
            if all_same_obs:
                color_patches.append(grey + (1,))
            elif type_min_obs in sensor_type_color_dict.keys():
                color_patches.append(sensor_type_color_dict[type_min_obs] + (1,))
            else:
                raise Exception("invalid type sensed!")
        # Sensor sectors
        color_patches = np.array(color_patches)
        coll_sensors = mc.PatchCollection(sensor_patches, facecolors=color_patches, edgecolors=almost_black, linewidth=1, alpha=0.5, zorder=1)
        ax.add_collection(coll_sensors)
        coll_list.append(coll_sensors)
        return tuple(coll_list)
    # Create animation
    ani = animation.FuncAnimation(fig, animate, len(env.state_history))
    ani.save(filename, fps=int(1/env.dt), extra_args=['-vcodec', 'libx264'])
    plt.close('all')

def _color(r, g, b):
    return (r/255, g/255, b/255)