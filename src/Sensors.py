"""

 Sensors.py  (author: Anson Wong / git: ankonzoid)

 External functions:
   sense, sense_sensors

 Internal functions:
   _make_sectors, _bins_included, _make_sector_observation


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
import sys
import numpy as np
from sympy import Point, Line

from SensorsBase import SensorsBase

class Sensors(SensorsBase):

    def __init__(self, settings):

        # Read settings
        self.n_sectors = settings.n_sectors
        self.sector_length = settings.sector_length  # make sure this is > radius_agent
        self.types_detect = settings.types_detect  # particles type to detect
        self.append_vxy_xy = settings.append_vxy_xy  # append (vx,vy,x,y) to observation?

        # Observation space
        self.sector_dim_size = len(self.types_detect)
        self.observation_size = self.sector_dim_size * self.n_sectors + self.append_vxy_xy.count(True)  # add velocity

    def sense(self, state, pid_agent, universe):
        # Sense environment
        sector_observation, sectors = self.sense_sensors(state, pid_agent, universe)

        # Make agent observations
        observation = list()  # flattened sensor observations
        for sublist in sector_observation:
            for item in sublist:
                observation.append(item)

        # Append the velocities
        state_observer = state[pid_agent,:]
        pind = universe.state_dict
        if self.append_vxy_xy[0] == True:
            observation.append(state_observer[pind["vx"]])  # append agent vx
        if self.append_vxy_xy[1] == True:
            observation.append(state_observer[pind["vy"]])  # append agent vy
        if self.append_vxy_xy[2] == True:
            observation.append(state_observer[pind["x"]])  # append agent x
        if self.append_vxy_xy[3] == True:
            observation.append(state_observer[pind["y"]])  # append agent y

        if len(observation) != self.observation_size:
            sys.exit("err: Inconsistent observation size!")

        # Update observation
        observation = np.array(observation, dtype=np.float)
        return observation

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

    # ==================================
    #
    # Internal functions
    #
    # ==================================

    def _make_sectors(self, n_sectors, length):
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

    def _make_sector_observation(self, xy_agent, r_agent, sectors, dtheta_sectors, sector_length,
               xy_particles, type_particles, r_particles, walls):

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
        sectors_temp.append(0.0)
        sectors_temp.insert(0, 0.0)

        wall_pts = []
        for wall in walls:
            wall_pts.append(wall[0])
        wall_pts = np.array(wall_pts)
        x_wall_min, x_wall_max = np.min(wall_pts[:, 1]), np.max(wall_pts[:, 1])
        y_wall_min, y_wall_max = np.min(wall_pts[:, 0]), np.max(wall_pts[:, 0])

        for j_sector in range(n_sectors):

            sector_obs_types = []
            for type_detect in types_detect:
                # Distance between agent and particles belonging to sector bin
                dist_D_sector_type = dist_D[(bin_counts[:,j_sector] == 1) & (type_D == type_detect)]
                if type_detect == 0:  # detect walls

                    # todo
                    sector_obs_type = sector_length
                    if xy_agent[0] + sector_length > x_wall_max:
                        sector_obs_type = x_wall_max - xy_agent[0]
                    elif xy_agent[0] - sector_length < x_wall_min:
                        sector_obs_type = xy_agent[0] - x_wall_min
                    elif xy_agent[1] + sector_length > y_wall_max:
                        sector_obs_type = x_wall_max - xy_agent[1]
                    elif xy_agent[1] - sector_length < y_wall_min:
                        sector_obs_type = xy_agent[1] - y_wall_min

                    sector_obs_type = (sector_obs_type - r_agent) / (sector_length - r_agent)

                elif len(dist_D_sector_type) == 0:  # no detection of particles

                    sector_obs_type = 1.0

                else:

                    # Observation measurement is the scaled distance of circe edges:
                    # 0.0 = when the agent and particle radii are touching edges
                    # 1.0 = furthest distance of agent sensor + particle detection
                    sector_obs_type = (np.amin(dist_D_sector_type) - r_particles - r_agent)/(sector_length - r_agent)

                sector_obs_types.append(sector_obs_type)
            sector_observation.append(sector_obs_types)

            # Sense walls

        print(sector_observation)
        exit()
        """
        obs_sectors = []
        for wall in walls:
            line_wall = Line(Point(wall[0][0], wall[0][1]), Point(wall[1][0], wall[1][1]))
            theta_1, theta_2 = sectors_temp[j_sector], sectors_temp[j_sector+1]
            sensor_1, sensor_2 = xy_agent.copy(), xy_agent.copy()
            sensor_1[0] += sector_length * np.cos(theta_1)
            sensor_1[1] += sector_length * np.sin(theta_1)
            sensor_2[0] += sector_length * np.cos(theta_2)
            sensor_2[1] += sector_length * np.sin(theta_2)
            line_sensor_1 = Line(Point(xy_agent[0], xy_agent[1]), Point(sensor_1[0], sensor_1[1]))
            line_sensor_2 = Line(Point(xy_agent[0], xy_agent[1]), Point(sensor_2[0], sensor_2[1]))

            int_1 = line_wall.intersection(line_sensor_1)
            int_2 = line_wall.intersection(line_sensor_2)

            obs = sector_length
            if len(int_1) > 0:
                int_1_pt = np.array([float(int_1[0][0]), float(int_1[0][1])])
                d = np.linalg.norm(int_1_pt - xy_agent)
                if d < obs: obs = d
            if len(int_2) > 0:
                int_2_pt = np.array([float(int_2[0][0]), float(int_2[0][1])])
                d = np.linalg.norm(int_2_pt - xy_agent)
                if d < obs: obs = d
            # convert obs to our formulation
            #obs = (obs - r_agent)/(sector_length - r_agent)
            obs_sectors.append(obs)
        print(obs_sectors)
        exit()
        """

        return sector_observation