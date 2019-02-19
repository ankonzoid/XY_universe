"""

 Universe.py  (author: Anson Wong / git: ankonzoid)

 External functions:
   reset, step, make_animation

 Internal functions:
   _get_walls


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
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as mc
from matplotlib.patches import Wedge

class Universe():

    def __init__(self, n_good, n_bad):

        # Name
        self.name = "survival"  # agent settings

        # Universe settings
        self.dt = 1.0/30.0
        self.mass = 0.05
        self.dv_agent, self.v_particles = 0.01, 0.05
        self.radius_agent, self.radius_particles = 0.05, 0.1
        self.box_half_length = 2.0
        self.reward_mode = "collect"  # "collect", "go_x"
        self.competition_mode = "none"  # "none", "coop", "compete"
        self.particle_mode = "none"  # "none", "chase_agent"
        self.add_wall_collisions = True
        self.add_particle_collisions = False
        self.add_agent_interactions = True
        self.state_init, self.pid_agents = self._init_state(n_good, n_bad)

        # Read settings
        self.n_agents = len(self.pid_agents)
        self.reward_mode = settings.reward_mode
        self.competition_mode = settings.competition_mode
        self.types_detect = settings.types_detect

        # State and action space
        self.state_dict = {"x": 0, "y": 1, "vx": 2, "vy": 3, "type": 4}
        self.action_name_dict = {0:"up", 1:"right", 2:"down", 3:"left"}
        self.action_size = len(self.action_name_dict.keys())

        # Frame
        self.box_dict = {"x_min": 0, "x_max": 1, "y_min": 2, "y_max": 3}
        self.walls = self._get_walls()

        # Make checks
        if self.state_init.shape[1] != len(self.state_dict):
            sys.exit("err: initial state has wrong dimensions!")

    def reset(self):
        state = self.state_init.copy()
        return state

    def step(self, state, action):
        actname = self.action_name_dict
        pind = self.state_dict
        (pind_x, pind_y) = (pind["x"], pind["y"])
        (pind_vx, pind_vy, pind_type) = (pind["vx"], pind["vy"], pind["type"])
        (pind_xy, pind_vxy) = ([pind_x,pind_y], [pind_vx,pind_vy])
        (xmin_box, xmax_box) = (-self.box_half_length, self.box_half_length)
        (ymin_box, ymax_box) = (-self.box_half_length, self.box_half_length)

        # Agent pids
        pid_agents = self.pid_agents  # agents do not disappear
        n_agents = self.n_agents

        # ===========================
        # Initialize output
        # ===========================
        state_next, rewards, dones = state.copy(), [0.0]*n_agents, [False]*n_agents

        # ===========================
        # Perform agent actions
        # ===========================
        if np.any(state_next[pid_agents, pind_type] >= 0):
            sys.exit("err: Trying to perform action on non-agent particle!")

        if actname[action] == "up":
            state_next[0, pind_vy] += self.dv_agent
        elif actname[action] == "right":
            state_next[0, pind_vx] += self.dv_agent
        elif actname[action] == "down":
            state_next[0, pind_vy] += -self.dv_agent
        elif actname[action] == "left":
            state_next[0, pind_vx] += -self.dv_agent
        else:
            sys.exit("Invalid action!")

        # ===========================
        # Perform particle actions
        # ===========================
        if self.particle_mode == "none":
            pass
        elif self.particle_mode == "chase_agent":  # all particles chase agent
            pid_chase = 0  # chase index
            state_next[n_agents:,pind_vxy] = state_next[pid_chase,pind_xy] - state_next[n_agents:,pind_xy]  # chase direction
            v_norm_particles = np.linalg.norm(state_next[n_agents:, pind_vxy], axis=1)  # norm of all velocities
            state_next[n_agents:,pind_vxy] = self.v_particles * np.divide(state_next[n_agents:,pind_vxy], v_norm_particles[:,None])
        else:
            sys.exit("err: Invalid particle mode of {}".format(self.particle_mode))

        # ======================================================
        #
        # Update positions:
        #  x = x + v * dt
        #
        # ======================================================
        state_next[:,pind_x] += state_next[:,pind_vx] * self.dt  # update x-coordinates
        state_next[:,pind_y] += state_next[:,pind_vy] * self.dt  # update y-coordinates

        # ======================================================
        # Add particle-particle collisions
        # ======================================================
        if self.add_particle_collisions:
            # Find the radius of every object
            radius_state = self.radius_particles * np.ones(len(state_next), dtype=np.float)
            radius_state[pid_agents] = self.radius_agent
            # Simular particle-particle collisions
            D = squareform(pdist(state_next[:, pind_xy]))  # N x N distance matrix
            C = np.add.outer(radius_state, radius_state)  # N x N criterion matrix
            plist1, plist2 = np.where(D < C)
            unique = (plist1 < plist2)  # unique p-p with ind1 < ind2
            plist1, plist2 = plist1[unique], plist2[unique]  # unique particle list of p-p
            for (p1, p2) in zip(plist1, plist2):  # update velocities of colliding particle pairs
                # Common parameters
                x1, x2 = state_next[p1, pind_xy], state_next[p2, pind_xy]
                v1, v2 = state_next[p1, pind_vxy], state_next[p2, pind_vxy]
                r1, r2, m1, m2 = radius_state[p1], radius_state[p2], self.mass, self.mass
                dist_12 = np.linalg.norm(x1 - x2)
                d_12_unit = (x1 - x2) / dist_12
                # Repositioning particles so they 'just' touch
                v1_normal = np.dot(v1, d_12_unit) * d_12_unit
                v2_normal = np.dot(v2, d_12_unit) * d_12_unit
                v1_tangent = v1 - v1_normal
                v2_tangent = v2 - v2_normal
                L = r1 + r2 - dist_12  # overlap parameter
                v12_normal_norm = np.linalg.norm(v1_normal - v2_normal)
                s_backwards = -L / v12_normal_norm
                x1 += s_backwards * v1_normal
                x2 += s_backwards * v2_normal
                # Perform particle collision equations after repositioning
                u1 = np.dot(v1_normal, d_12_unit)
                u2 = np.dot(v2_normal, d_12_unit)
                v1_normal = (((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2)) * d_12_unit
                v2_normal = (((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2)) * d_12_unit
                v1 = v1_normal + v1_tangent
                v2 = v2_normal + v2_tangent
                # Update state
                state_next[p1, pind_xy] = x1
                state_next[p2, pind_xy] = x2
                state_next[p1, pind_vxy] = v1
                state_next[p2, pind_vxy] = v2

        # ======================================================
        # Add particle-wall interactions
        # ======================================================
        if self.add_wall_collisions:
            # Find the radius of every object
            radius_state = self.radius_particles * np.ones(len(state_next), dtype=np.float)
            radius_state[pid_agents] = self.radius_agent
            # Find which particles/agents crossed the wall (they need to be rebounded back in)
            crossed_x1 = (state_next[:,pind_x] < xmin_box + radius_state[:])
            crossed_x2 = (state_next[:,pind_x] > xmax_box - radius_state[:])
            crossed_y1 = (state_next[:,pind_y] < ymin_box + radius_state[:])
            crossed_y2 = (state_next[:,pind_y] > ymax_box - radius_state[:])
            # Update locations after wall collision
            state_next[crossed_x1,pind_x] = xmin_box + radius_state[crossed_x1]
            state_next[crossed_x2,pind_x] = xmax_box - radius_state[crossed_x2]
            state_next[crossed_y1,pind_y] = ymin_box + radius_state[crossed_y1]
            state_next[crossed_y2,pind_y] = ymax_box - radius_state[crossed_y2]
            # Update velocities after wall collision (flip velocities using momentum equations)
            state_next[crossed_x1|crossed_x2,pind_vx] *= -1
            state_next[crossed_y1|crossed_y2,pind_vy] *= -1

        # ======================================================
        # Add agent-particle interactions (gives rewards)
        # ======================================================
        if self.add_agent_interactions:
            # Find captured particles
            for i, pid_agent in enumerate(pid_agents):
                # Capture particles and delete them
                type_agent = state_next[pid_agent, pind_type]
                xy_particles_relative = state_next[n_agents:,pind_xy] - state_next[pid_agent,pind_xy]
                dist_particles = np.linalg.norm(xy_particles_relative, axis=1)  # distance between agent-particles
                pids_captured = np.where(dist_particles < (self.radius_agent + self.radius_particles))[0] + n_agents
                types_captured = np.array(state_next[pids_captured,pind_type], dtype=np.int)
                n_type_captured = [0, int(np.sum(types_captured==1)), int(np.sum(types_captured==2))]
                state_next = np.delete(state_next, pids_captured, axis=0)

                # Compute reward (type 1 = positive, type 2 = negative)
                if self.reward_mode == "collect":
                    rewards[i] += 100 * n_type_captured[1]  # reward for collection
                    rewards[i] += 1  # reward for surviving
                    dones[i] = (n_type_captured[2] > 0)

                elif self.reward_mode == "go_x":
                    dx_scaled = (state_next[pid_agent, pind_x] - state[pid_agent, pind_x])/(2*self.box_half_length)
                    if type_agent == -1:  # go right
                        rewards[i] += 2000 * (dx_scaled)
                        rewards[i] += -1  # penalize for existence
                        dones[i] = (state_next[pid_agent, pind_x] > 0.85 * self.box_half_length)
                    elif type_agent == -2:  # go left
                        rewards[i] += 2000 * (-dx_scaled)
                        rewards[i] += -1  # penalize for existence
                        dones[i] = (state_next[pid_agent, pind_x] < -0.85 * self.box_half_length)
                    else:
                        sys.exit("err: Invalid agent type for our reward_mode = {}".format(self.reward_mode))

                elif self.reward_mode == "passby_x":
                    dx_scaled = (state_next[pid_agent, pind_x] - state[pid_agent, pind_x])/(2*self.box_half_length)
                    if type_agent == -1:  # go right
                        rewards[i] += 2000 * (dx_scaled)
                        rewards[i] += -1  # penalize for existence
                        dones[i] = (state_next[pid_agent, pind_x] > 0.85 * self.box_half_length)
                    elif type_agent == -2:  # go left
                        rewards[i] += 2000 * (-dx_scaled)
                        rewards[i] += -1  # penalize for existence
                        dones[i] = (state_next[pid_agent, pind_x] < -0.85 * self.box_half_length)
                    else:
                        sys.exit("err: Invalid agent type for our reward_mode = {}".format(self.reward_mode))

                else:
                    sys.exit("err: No reward scheme given!")

            # Add competition explicitly
            if self.competition_mode == "compete":
                rewards_compete = rewards.copy()
                for i, pid_agent in enumerate(pid_agents):
                    rewards_others = rewards[:i] + rewards[i+1:]
                    rewards_compete[i] -= sum(rewards_others)
                rewards = rewards_compete

            elif self.competition_mode == "coop":
                rewards_coop = rewards.copy()
                for i, pid_agent in enumerate(pid_agents):
                    rewards_others = rewards[:i] + rewards[i+1:]
                    rewards_coop[i] += sum(rewards_others)
                rewards = rewards_coop

            elif self.competition_mode == "none":
                pass

            else:
                sys.exit("err: No competition mode given!")

        return state_next, rewards, dones

    def make_animation(self, states, universe, sensors, save_ani=False, filename='particleworld.mp4'):
        fig, ax = plt.subplots(tight_layout=True)

        sector_length = sensors.sector_length
        radius_agent = self.radius_agent
        radius_particles = self.radius_particles
        box_half_length = self.box_half_length
        pind_x = self.state_dict["x"]
        pind_y = self.state_dict["y"]
        pind_xy = [pind_x, pind_y]
        pind_type = self.state_dict["type"]

        pid_agents = self.pid_agents

        # Animation bounds
        x_min, x_max = -box_half_length, box_half_length
        y_min, y_max = -box_half_length, box_half_length

        # Color definitions
        def _color(r, g, b):
            return (r/255, g/255, b/255)

        orange, light_orange = _color(255, 153, 51), _color(244, 164, 96)
        brown, light_brown = _color(198, 74, 12), _color(222, 184, 135)
        dark_red, red = _color(220, 20, 60), _color(255, 153, 153)
        dark_blue, blue = _color(65, 105, 225), _color(153, 204, 255)
        dark_grey, grey = _color(89, 89, 89), _color(169, 169, 169)
        almost_black = _color(30, 30, 30)

        # Color dictionary
        type_color_dict = {-2: brown, -1: orange, 0: dark_grey, 1: dark_red, 2: dark_blue}
        sensor_type_color_dict = {-2: light_brown, -1: light_orange, 0: dark_grey, 1: red, 2: blue}
        if type_color_dict.keys() != sensor_type_color_dict.keys():
            sys.exit("err: Non-matching keys for our particle and sensed color dictionaries!")

        # Axes settings
        ax.margins(0.1, tight=True)

        # Set aspect ratio to be square
        ax.set_aspect(abs(x_max-x_min)/abs(y_max-y_min))

        # Draw circles
        def _get_circles_particle_type(x_particles, type, radius):
            idx = np.where(np.array(x_particles[:,pind_type], dtype=np.int) == type)[0]
            centers = x_particles[idx][:,pind_xy]
            radii = radius * np.ones(len(idx), dtype=np.float)
            return (centers, radii)

        # Start
        def animate(i):
            ax.clear()  # clear up objects that were previously added
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # =================================
            # Draw particles (all types)
            # =================================
            coll_list = []

            for type in type_color_dict:
                facecolor = type_color_dict[type]
                zorder = 3 if type < 0 else 2
                radius_draw = radius_agent if type < 0 else radius_particles

                (centers, radii) = _get_circles_particle_type(states[i], type, radius_draw)  # good: type 1, red
                patches = [plt.Circle(center, radius) for (center, radius) in zip(centers, radii)]
                coll = mc.PatchCollection(patches, facecolors=facecolor, edgecolors=almost_black, linewidth=1, zorder=zorder)
                ax.add_collection(coll)
                coll_list.append(coll)

            # =================================
            # Draw sensors
            # =================================
            for i_agent, pid_agent in enumerate(pid_agents):
                # Make observation
                state_agent = states[i][pid_agent]
                sector_observation, sectors = sensors.sense_sensors(states[i], pid_agent, universe)

                # Draw sectors
                sensor_patches, color_patches = list(), list()
                for j, (sector_obs, sector) in enumerate(zip(sector_observation, sectors)):

                    # Type of wedge (based on which one has lowest sector_obs)
                    all_same_obs = all(x == 1.0 for x in sector_obs)  # check if all == 1
                    idx_type_min_obs = sector_obs.index(min(sector_obs))
                    type_min_obs = self.types_detect[idx_type_min_obs]

                    # Set theta (in degrees)
                    radians_to_degrees = 180 / np.pi
                    if j == 0:
                        theta_1 = 0 * radians_to_degrees
                        theta_2 = sectors[j] * radians_to_degrees
                    else:
                        theta_1 = sectors[j-1] * radians_to_degrees
                        theta_2 = sectors[j] * radians_to_degrees

                    # Draw wedge
                    sensor_obs = (sector_obs[idx_type_min_obs] * (sector_length - radius_agent)) + radius_agent
                    obs_sensor_length = sensor_obs if sensor_obs >= 0 else 0.0
                    wedge = Wedge((state_agent[pind_x], state_agent[pind_y]), obs_sensor_length, theta_1, theta_2, alpha=0.5)

                    # Append wedge shape
                    sensor_patches.append(wedge)

                    # Append wedge color
                    if all_same_obs:
                        color_patches.append(grey + (1,))
                    elif type_min_obs in sensor_type_color_dict.keys():
                        color_patches.append(sensor_type_color_dict[type_min_obs] + (1,))
                    else:
                        sys.exit("err: Invalid type sensed!")

                # Sensor sectors
                color_patches = np.array(color_patches)
                coll_sensors = mc.PatchCollection(sensor_patches, facecolors=color_patches, edgecolors=almost_black, linewidth=1, alpha=0.5, zorder=1)
                ax.add_collection(coll_sensors)
                coll_list.append(coll_sensors)

            return tuple(coll_list)

        # Create animation
        ani = animation.FuncAnimation(fig, animate, len(states))

        # Save or show animation
        if save_ani:
            ani.save(filename, fps=int(1/self.dt), extra_args=['-vcodec', 'libx264'])
        else:
            plt.show()
        plt.close('all')

    # ==================================
    #
    # Internal functions
    #
    # ==================================

    def _get_walls(self):
        corner_1 = np.array([-self.box_half_length, -self.box_half_length])
        corner_2 = np.array([-self.box_half_length, self.box_half_length])
        corner_3 = np.array([self.box_half_length, self.box_half_length])
        corner_4 = np.array([self.box_half_length, -self.box_half_length])
        walls = [(corner_1, corner_2), (corner_2, corner_3),
                 (corner_3, corner_4), (corner_4, corner_1)]
        return walls

    def _init_state(self, n_good, n_bad, seed=0):
        np.random.seed(seed)
        pind = {"x": 0, "y": 1, "vx": 2, "vy": 3, "type": 4}
        (pind_xy, pind_vxy) = ([pind["x"], pind["y"]], [pind["vx"], pind["vy"]])
        (pind_vx, pind_vy, pind_type) = (pind["vx"], pind["vy"], pind["type"])

        # Define initial state
        idx = 0
        n_type_agents = (1, 0)  # type -1, type -2
        n_type_particles = (n_good, n_bad)  # type 1 (good), type 2 (bad)
        #n_type_particles = (0, 110)  # type 1 (good), type 2 (bad)
        #n_type_particles = (55, 55)  # type 1 (good), type 2 (bad)
        n, d = sum(list(n_type_agents)) + sum(list(n_type_particles)), len(pind)  # everything
        # Random state
        state = -0.5 + np.random.random((n, d))  # random numbers from standard normal distribution
        state[:,pind_xy] *= 4.0  # spread out particle positions
        for i, n_type in enumerate(n_type_agents):
            state[idx:(idx+n_type),pind_type] = -(i+1)
            idx += n_type
        for i, n_type in enumerate(n_type_particles):
            state[idx:(idx+n_type),pind_type] = (i+1)
            idx += n_type

        # Make check
        if idx != n:
            sys.exit("err: idx is not equal to n!")

        # Find pids
        n_total = state.shape[0]
        pid_agents = np.where(state[:,pind_type] < 0)[0]
        n_agents = len(pid_agents)
        n_particles = n_total - n_agents

        # Set particle velocities
        if n_particles > 0:
            v_norm = np.linalg.norm(state[n_agents:,pind_vxy], axis=1)  # velocity magnitudes
            state[n_agents:,pind_vxy] = self.v_particles * np.divide(state[n_agents:,pind_vxy], v_norm[:, None])

        # Set agent velocities
        if n_agents > 0:
            state[:n_agents,pind_vx], state[:n_agents,pind_vy] = 0.0, 0.0  # set zero agent velocities

        # Reset seed randomly
        np.random.seed(seed=None)
        return state, pid_agents