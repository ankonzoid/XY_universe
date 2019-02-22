"""

 Sensors.py  (author: Anson Wong / git: ankonzoid)


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

class Sensors():

    def __init__(self, n_sectors, sector_radius):

        # Settings
        self.n_sectors = n_sectors
        self.sector_length = sector_radius # make sure this is > radius_agent
        self.types_detect = [0, 1, 2]  # particle types (1=good, 2=bad, -1/-2=agents, 0=walls)
        self.append_vxy_xy = [True, True, False, False] # append (vx,vy,x,y) to observation?

        # Observation space
        self.sector_dim_size = len(self.types_detect)
        self.observation_size = self.sector_dim_size * self.n_sectors + \
                                self.append_vxy_xy.count(True)  # add velocity

    def sense_sensors(self, state, pid_agent, universe):
        n_agents = universe.n_agents
        r_agent = universe.radius_agent
        r_particles = universe.radius_particles
        walls = universe.walls
        pind_xy = [universe.state_dict["x"], universe.state_dict["y"]]
        pind_type = universe.state_dict["type"]

        xy_agent = state[pid_agent, pind_xy]
        xy_particles = np.delete(state[:, pind_xy], [pid_agent], axis=0)
        type_particles = np.delete(state[:, pind_type], [pid_agent], axis=0)

        # Sensor parameters
        n_sectors = self.n_sectors
        sector_length = self.sector_length

        sectors, dtheta_sectors = self._make_sectors(n_sectors, sector_length)
        sector_observation = self._make_sector_observation(xy_agent, r_agent, sectors, dtheta_sectors, sector_length,
                                    xy_particles, type_particles, r_particles, walls)

        return sector_observation, sectors

    # Sense
    def sense(self, state, pid_agent, env):

        # Sense environment
        sector_observation, sectors = self.sense_sensors(state, pid_agent, env)

        # Make agent observations
        obs = list()  # flattened sensor observations
        for sublist in sector_observation:
            for item in sublist:
                obs.append(item)

        # Append the velocities
        state_obs = state[0, :]
        pidx = env.state_dict
        if self.append_vxy_xy[0] == True:
            obs.append(state_obs[pidx["vx"]])  # append agent vx
        if self.append_vxy_xy[1] == True:
            obs.append(state_obs[pidx["vy"]])  # append agent vy
        if self.append_vxy_xy[2] == True:
            obs.append(state_obs[pidx["x"]])  # append agent x
        if self.append_vxy_xy[3] == True:
            obs.append(state_obs[pidx["y"]])  # append agent y

        if len(obs) != self.observation_size:
            sys.exit("err: Inconsistent observation size!")

        # Update observation
        obs = np.array(obs, dtype=np.float)
        return obs, sectors

    # ==================================
    #
    # Internal functions
    #
    # ==================================

    def _make_sectors(self, n_sectors, sector_length):
        # Define the sectors by a scalar theta of its maximum value
        # Note: theta is counter-clockwise away from positive x-axis
        theta_range = 2*np.pi
        dtheta_sectors = theta_range / n_sectors
        sectors = list()
        for i in range(n_sectors):
            theta_2 = (i+1) * dtheta_sectors
            sectors.append(theta_2)
        return sectors, dtheta_sectors

    def _bins_included(self, idx_bin_center, idx_bin_min, idx_bin_max, n_bins):
        # Find the bins within the min and max bin indices
        n_bins_plus = (idx_bin_max - idx_bin_center) % n_bins  # number of bins + from center
        n_bins_minus = (idx_bin_center - idx_bin_min) % n_bins  # number of bin - from center
        idx_bins = [idx_bin_center]
        for i in range(n_bins_plus):
            idx_bins.append((idx_bin_center + (i+1)) % n_bins)
        for i in range(n_bins_minus):
            idx_bins.append((idx_bin_center - (i+1)) % n_bins)
        return idx_bins

    def _make_sector_observation(self, xy_agent,
                                 r_agent, sectors,
                                 dtheta_sectors, sector_length,
                                 xy_particles, type_particles,
                                 r_particles, walls):

        # Compute distances of all particles from agent
        xy_particles_relative = xy_particles - xy_agent  # particle relative to agent
        dist = np.linalg.norm(xy_particles_relative, axis=1)  # find distances
        i_particles_D = np.where(dist < (sector_length + r_particles))[0]  # particles indices within sensor length

        # Angles are relative to positive x-axis direction
        twopi = 2 * np.pi
        type_D = type_particles[i_particles_D]
        dist_D = dist[i_particles_D]
        xy_D = xy_particles_relative[i_particles_D]

        # Compute the angle subtended by agent-center with particle-edges
        r_particles_D = r_particles * np.ones(len(dist_D), dtype=np.float)
        dtheta_particles_D = np.arctan2(r_particles_D[:], dist_D[:])  # angle of agent-center with particle-edges

        # Compute angle of agent-center with particle-center (and particle edges)
        theta_particles_D = np.arctan2(xy_D[:,1], xy_D[:,0]) % twopi  # angle of agent-center and particle-center
        theta_max_particles_D = (theta_particles_D + dtheta_particles_D) % twopi
        theta_min_particles_D = (theta_particles_D - dtheta_particles_D) % twopi

        # Create bins for sector angles (for 3 bins you would have bins = [2*pi/3, 4*pi/3])
        bins = np.array(sectors[:-1])  # bins

        # Find which bins
        sector_bins_D = np.digitize(theta_particles_D, bins)  # values, bins
        sector_bins_max_D = np.digitize(theta_max_particles_D, bins)
        sector_bins_min_D = np.digitize(theta_min_particles_D, bins)

        # Set up the bin counts
        n_bins = len(sectors)
        n_particles_D = len(i_particles_D)

        bin_counts = np.zeros((n_particles_D, n_bins), dtype=np.int)
        for idx_particle, (bin_center, bin_min, bin_max) in enumerate(zip(sector_bins_D, sector_bins_min_D, sector_bins_max_D)):
            idx_bins = self._bins_included(bin_center, bin_min, bin_max, n_bins)
            bin_counts[idx_particle, idx_bins] = 1

        # Find closes point in each bin
        types_detect = self.types_detect
        sector_observation = list()

        n_sectors = len(sectors)
        sectors_temp = sectors.copy()
        sectors_temp.insert(0, 0.0)

        for j_sector in range(n_sectors):

            sector_obs_types = []
            for type_detect in types_detect:

                # Distance between agent and particles belonging to sector bin
                sector_obs_type = 1.0  # no detection of particles (default)
                if type_detect == 0:

                    # Sensor end points
                    theta_left = sectors_temp[j_sector]
                    theta_right = sectors_temp[j_sector + 1]
                    theta_middle = (theta_left + theta_right)/2
                    sensor_left = np.array([xy_agent[0] + sector_length * np.cos(theta_left),
                                            xy_agent[1] + sector_length * np.sin(theta_left)])
                    sensor_right = np.array([xy_agent[0] + sector_length * np.cos(theta_right),
                                             xy_agent[1] + sector_length * np.sin(theta_right)])
                    sensor_middle = np.array([xy_agent[0] + sector_length * np.cos(theta_middle),
                                              xy_agent[1] + sector_length * np.sin(theta_middle)])

                    # Find shortest distances
                    distances = []
                    for wall in walls:
                        result_left = self._line_intersection(xy_agent, sensor_left, wall[0], wall[1])
                        result_right = self._line_intersection(xy_agent, sensor_right, wall[0], wall[1])
                        result_middle = self._line_intersection(xy_agent, sensor_middle, wall[0], wall[1])
                        if result_left[2] == 1 and result_left[3] >= 0.0 and result_left[3] <= 1.0:
                            d = np.linalg.norm(np.array([result_left[0], result_left[1]]) - xy_agent)
                            distances.append(d)
                        if result_right[2] == 1 and result_right[3] >= 0.0 and result_right[3] <= 1.0:
                            d = np.linalg.norm(np.array([result_right[0], result_right[1]]) - xy_agent)
                            distances.append(d)
                        if result_middle[2] == 1 and result_middle[3] >= 0.0 and result_middle[3] <= 1.0:
                            d = np.linalg.norm(np.array([result_middle[0], result_middle[1]]) - xy_agent)
                            distances.append(d)

                    if len(distances) > 0:
                        obs_physical = np.amin(distances)
                        if(obs_physical < 0): exit("err: obs_physical < 0!")
                        sector_obs_type = (obs_physical - r_agent) / (sector_length - r_agent)

                else:

                    dist_D_sector_type = dist_D[(bin_counts[:, j_sector] == 1) & (type_D == type_detect)]
                    if len(dist_D_sector_type) > 0:
                        # Observation measurement is the scaled distance of circe edges:
                        # 0.0 = when the agent and particle radii are touching edges
                        # 1.0 = furthest distance of agent sensor + particle detection
                        obs_physical = np.amin(dist_D_sector_type) - r_particles
                        sector_obs_type = (obs_physical - r_agent) / (sector_length - r_agent)

                sector_obs_types.append(sector_obs_type)
            sector_observation.append(sector_obs_types)

        return sector_observation

    def _line_intersection(self, ptA1, ptA2, ptB1, ptB2):
        DET_TOLERANCE = 0.00000001
        # the first line is ptA1 + r*(ptA2 - ptA1)
        A1_x, A1_y = ptA1
        A2_x, A2_y = ptA2
        dx_A, dy_A = A2_x - A1_x, A2_y - A1_y
        # the second line is ptB1 + s*(ptB2 - ptB1)
        B1_x, B1_y = ptB1
        B2_x, B2_y = ptB2
        dx_B, dy_B = B2_x - B1_x, B2_y - B1_y
        DET = (-dx_A * dy_B + dy_A * dx_B) # compute determinant
        if np.abs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0) # parallel lines if DET small
        DETinv = 1.0 / DET  # now, the determinant should be OK
        r = DETinv * (-dy_B * (B1_x - A1_x) + dx_B * (B1_y - A1_y))  # scalar along "self" segment
        s = DETinv * (-dy_A * (B1_x - A1_x) + dx_A * (B1_y - A1_y)) # scalar along input line
        xi = (A1_x + r * dx_A + B1_x + s * dx_B) / 2.0 # return description average
        yi = (A1_y + r * dy_A + B1_y + s * dy_B) / 2.0 # return description average
        return (xi, yi, 1, r, s)