"""

 Sensors.py  (author: Anson Wong / git: ankonzoid)

"""
import sys
import numpy as np

class Sensors():

    def __init__(self, n_sectors, sector_radius):
        self.n_sectors = n_sectors
        self.sector_length = sector_radius # make sure this is > radius_agent
        self.types_detect = [0, 1, 2]  # particle types to detect (1=good, 2=bad, -1=agent, 0=walls)
        self.append_vxy_xy = [True, True, False, False] # append (vx,vy,x,y) to observation?
        self.sector_dim_size = len(self.types_detect)
        self.observation_size = self.sector_dim_size * self.n_sectors + self.append_vxy_xy.count(True)

    def sense(self, env, force_sense_custom_state=False, custom_state=None):
        if force_sense_custom_state and custom_state is not None:
            xy_agent = custom_state[0, [env.state_dict["x"], env.state_dict["y"]]]  # agent xy
            xy_particles = np.delete(custom_state[:, [env.state_dict["x"], env.state_dict["y"]]], [0], axis=0)  # background particles xy
            type_particles = np.delete(custom_state[:, env.state_dict["type"]], [0], axis=0)  # background particle type
        else:
            xy_agent = env.state[0, [env.state_dict["x"], env.state_dict["y"]]]  # agent xy
            xy_particles = np.delete(env.state[:, [env.state_dict["x"], env.state_dict["y"]]], [0], axis=0)  # background particles xy
            type_particles = np.delete(env.state[:, env.state_dict["type"]], [0], axis=0)  # background particle type
        # Define the sectors by a scalar theta of its maximum value
        # Note: theta is counter-clockwise away from positive x-axis
        dtheta_sectors = (2 * np.pi) / self.n_sectors
        sectors = [(i + 1) * dtheta_sectors for i in range(self.n_sectors)]
        # Compute distances of all particles from agent
        xy_particles_relative = xy_particles - xy_agent  # particle relative to agent
        dist = np.linalg.norm(xy_particles_relative, axis=1)  # find distances
        i_particles_D = np.where(dist < (self.sector_length + env.radius_particles))[0]  # particles indices within sensor length
        # Angles are relative to positive x-axis direction
        type_D = type_particles[i_particles_D]
        dist_D = dist[i_particles_D]
        xy_D = xy_particles_relative[i_particles_D]
        # Compute the angle subtended by agent-center with particle-edges
        r_particles_D = env.radius_particles * np.ones(len(dist_D), dtype=np.float)
        dtheta_particles_D = np.arctan2(r_particles_D[:], dist_D[:])  # angle of agent-center with particle-edges
        # Compute angle of agent-center with particle-center (and particle edges)
        theta_particles_D = np.arctan2(xy_D[:, 1], xy_D[:, 0]) % (2 * np.pi)  # angle of agent-center and particle-center
        theta_max_particles_D = (theta_particles_D + dtheta_particles_D) % (2 * np.pi)
        theta_min_particles_D = (theta_particles_D - dtheta_particles_D) % (2 * np.pi)
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
            # Find the bins within the min and max bin indices
            n_bins_plus = (bin_max - bin_center) % n_bins  # number of bins + from center
            n_bins_minus = (bin_center - bin_min) % n_bins  # number of bin - from center
            idx_bins = [bin_center]
            for i in range(n_bins_plus):
                idx_bins.append((bin_center + (i + 1)) % n_bins)
            for i in range(n_bins_minus):
                idx_bins.append((bin_center - (i + 1)) % n_bins)
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
                    theta_middle = (theta_left + theta_right) / 2
                    sensor_left = np.array([xy_agent[0] + self.sector_length * np.cos(theta_left),
                                            xy_agent[1] + self.sector_length * np.sin(theta_left)])
                    sensor_right = np.array([xy_agent[0] + self.sector_length * np.cos(theta_right),
                                             xy_agent[1] + self.sector_length * np.sin(theta_right)])
                    sensor_middle = np.array([xy_agent[0] + self.sector_length * np.cos(theta_middle),
                                              xy_agent[1] + self.sector_length * np.sin(theta_middle)])
                    # Find shortest distances
                    distances = []
                    for wall in env.walls:
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
                        if (obs_physical < 0): exit("err: obs_physical < 0!")
                        sector_obs_type = (obs_physical - env.radius_agent) / (self.sector_length - env.radius_agent)
                else:
                    dist_D_sector_type = dist_D[(bin_counts[:, j_sector] == 1) & (type_D == type_detect)]
                    if len(dist_D_sector_type) > 0:
                        # Observation measurement is the scaled distance of circe edges:
                        # 0.0 = when the agent and particle radii are touching edges
                        # 1.0 = furthest distance of agent sensor + particle detection
                        obs_physical = np.amin(dist_D_sector_type) - env.radius_particles
                        sector_obs_type = (obs_physical - env.radius_agent) / (self.sector_length - env.radius_agent)
                sector_obs_types.append(sector_obs_type)
            sector_observation.append(sector_obs_types)
        # Make agent observations
        ob = []  # flattened sensor observations
        for sublist in sector_observation:
            ob.extend([item for item in sublist])
        if self.append_vxy_xy[0] == True:
            ob.append(env.state[0, env.state_dict["vx"]])  # append agent vx
        if self.append_vxy_xy[1] == True:
            ob.append(env.state[0, env.state_dict["vy"]])  # append agent vy
        if self.append_vxy_xy[2] == True:
            ob.append(env.state[0, env.state_dict["x"]])  # append agent x
        if self.append_vxy_xy[3] == True:
            ob.append(env.state[0, env.state_dict["y"]])  # append agent y
        if len(ob) != self.observation_size:
            sys.exit("err: Inconsistent observation size!")
        ob = np.array(ob, dtype=np.float)
        return ob, sector_observation, sectors

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