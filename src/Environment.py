"""

 Environment.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np

class Environment():

    def __init__(self, n_good, n_bad):
        self.state_dict = {"x": 0, "y": 1, "vx": 2, "vy": 3, "type": 4}
        self.action_name_dict = {0: "up", 1: "right", 2: "down", 3: "left"}
        self.action_size = len(self.action_name_dict.keys())
        self.dt = 1.0/30.0
        self.mass = 0.05
        self.dv_agent, self.v_particles = 0.01, 0.05
        self.radius_agent, self.radius_particles = 0.05, 0.1
        self.box_half_length = 2.0
        self.add_wall_collisions = True
        self.add_agent_particle_interactions = True
        self.state_init = self._init_state(n_good, n_bad)
        self.types_detect = [0, 1, 2]
        # Frame
        self.box_dict = {"x_min": 0, "x_max": 1, "y_min": 2, "y_max": 3}
        self.walls = self._get_walls()
        # Make checks
        if self.state_init.shape[1] != len(self.state_dict):
            raise Exception("initial state has wrong dimensions!")
        self.reset()

    def reset(self):
        self.state = self.state_init.copy()
        self.reward_history = [] # rewards of episode
        self.state_history = [] # states of episode
        return self.state

    def step(self, action):
        j_x, j_y = self.state_dict["x"], self.state_dict["y"]
        j_vx, j_vy, j_type = self.state_dict["vx"], self.state_dict["vy"], self.state_dict["type"]
        j_xy, j_vxy = [j_x, j_y], [j_vx, j_vy]
        # Build next state
        state_next, reward, done = self.state.copy(), 0.0, False
        if state_next[0, j_type] >= 0: # check agent is of type < 0
            raise Exception("err: Trying to perform action on non-agent particle!")
        if self.action_name_dict[action] == "up":
            state_next[0, j_vy] += self.dv_agent
        elif self.action_name_dict[action] == "right":
            state_next[0, j_vx] += self.dv_agent
        elif self.action_name_dict[action] == "down":
            state_next[0, j_vy] += -self.dv_agent
        elif self.action_name_dict[action] == "left":
            state_next[0, j_vx] += -self.dv_agent
        else:
            raise Exception("invalid action!")
        # Update positions: x = x + v * dt
        state_next[:, j_x] += state_next[:, j_vx] * self.dt
        state_next[:, j_y] += state_next[:, j_vy] * self.dt
        # Add wall interactions
        if self.add_wall_collisions:
            # Box wall locations
            xmin_box, xmax_box = -self.box_half_length, self.box_half_length
            ymin_box, ymax_box = -self.box_half_length, self.box_half_length
            # Find the radius of every object
            radius_state = self.radius_particles * np.ones(len(state_next), dtype=np.float)
            radius_state[0] = self.radius_agent
            # Find which particles/agents crossed the wall (they need to be rebounded back in)
            crossed_x1 = (state_next[:, j_x] < xmin_box + radius_state[:])
            crossed_x2 = (state_next[:, j_x] > xmax_box - radius_state[:])
            crossed_y1 = (state_next[:, j_y] < ymin_box + radius_state[:])
            crossed_y2 = (state_next[:, j_y] > ymax_box - radius_state[:])
            # Update locations after wall collision
            state_next[crossed_x1, j_x] = xmin_box + radius_state[crossed_x1]
            state_next[crossed_x2, j_x] = xmax_box - radius_state[crossed_x2]
            state_next[crossed_y1, j_y] = ymin_box + radius_state[crossed_y1]
            state_next[crossed_y2, j_y] = ymax_box - radius_state[crossed_y2]
            # Update velocities after wall collision (flip velocities using momentum equations)
            state_next[crossed_x1|crossed_x2, j_vx] *= -1
            state_next[crossed_y1|crossed_y2, j_vy] *= -1
        # Add agent-particle interactions (gives rewards)
        if self.add_agent_particle_interactions:
            xy_particles_relative = state_next[1:, j_xy] - state_next[0, j_xy]
            dist_particles = np.linalg.norm(xy_particles_relative, axis=1)  # distance between agent-particles
            idxs_captured = np.where(dist_particles < (self.radius_agent + self.radius_particles))[0] + 1
            types_captured = np.array(state_next[idxs_captured, j_type], dtype=np.int)
            n_type_captured = [0, int(np.sum(types_captured==1)), int(np.sum(types_captured==2))]
            state_next = np.delete(state_next, idxs_captured, axis=0)
            reward += 100 * n_type_captured[1]  # reward for collection
            reward += 1  # reward for surviving
            done = (n_type_captured[2] > 0)
        self.state = state_next # evolve
        self.reward_history.append(reward) # track reward
        self.state_history.append(state_next) # track states
        return state_next, reward, done

    def _get_walls(self):
        corner_1 = np.array([-self.box_half_length, -self.box_half_length])
        corner_2 = np.array([-self.box_half_length, self.box_half_length])
        corner_3 = np.array([self.box_half_length, self.box_half_length])
        corner_4 = np.array([self.box_half_length, -self.box_half_length])
        walls = [(corner_1, corner_2), (corner_2, corner_3), (corner_3, corner_4), (corner_4, corner_1)]
        return walls

    def _init_state(self, n_good, n_bad):
        j_xy, j_vxy = [self.state_dict["x"], self.state_dict["y"]], [self.state_dict["vx"], self.state_dict["vy"]]
        j_vx, j_vy, j_type = self.state_dict["vx"], self.state_dict["vy"], self.state_dict["type"]
        state = -0.5 + np.random.random((1 + n_good + n_bad, len(self.state_dict.keys()))) # initialize
        state[:, j_xy] *= 4.0  # spread out particle positions
        state[0, j_type] = -1 # agent type = -1
        state[1:(1 + n_good), j_type] = 1 # particle type 1
        state[(1 + n_good):(1 + n_good + n_bad), j_type] = 2 # particle type 2
        if n_good + n_bad > 0:
            v_norm = np.linalg.norm(state[1:, j_vxy], axis=1)  # velocity magnitudes
            state[1:, j_vxy] = self.v_particles * np.divide(state[1:, j_vxy], v_norm[:, None])
        state[:1, j_vx] = 0.0 # zero vx
        state[:1, j_vy] = 0.0 # zero vy
        return state