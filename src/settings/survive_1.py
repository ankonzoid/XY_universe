"""

 survive_1.py  (author: Anson Wong / git: ankonzoid)


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

class Settings:

    def __init__(self):
        # Name
        self.name = "survive_1"  # agent settings

        # Run settings
        self.n_episodes = 500
        self.n_iter = 2000

        # Config settings
        self.save_settings, self.save_log = True, True
        self.save_model, self.save_animation = True, True
        self.reward_threshold_save = 2000  # log reward threshold passing
        self.episodes_reward_threshold_save = 10  # num episode of threshold log for animation save

        # Agent settings
        self.epsilon_range, self.epsilon_decay = [0.2, 0.2], 1.0
        self.gamma = 0.99  # discount
        self.hidden_layers = [100, 100]
        self.L2_regularizer = [1E-5, 1E-5]
        self.learning_rate, self.activation, self.loss, self.n_epochs = 0.005, "relu", "mse", 1
        self.memory_size, self.training_batch_size = 10000, 200
        self.use_target_model = False
        self.n_training_sessions = 20

        # Universe settings
        self.dt = 1/30
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
        self.state_init, self.pid_agents = self._init_state()

        # Sensors settings
        self.n_sectors = 4
        self.sector_length = 1.0
        self.types_detect = [0, 2]  # 1=good, 2=bad, -1/-2=agents, 0=walls
        self.append_vxy_xy = [True, True, False, False]

    def _init_state(self, seed=0):
        np.random.seed(seed)
        pind = {"x": 0, "y": 1, "vx": 2, "vy": 3, "type": 4}
        (pind_xy, pind_vxy) = ([pind["x"], pind["y"]], [pind["vx"], pind["vy"]])
        (pind_vx, pind_vy, pind_type) = (pind["vx"], pind["vy"], pind["type"])

        # Define initial state
        if 0:
            idx = 0
            n_type_agents = (1, 1)  # type -1, type -2
            n_type_particles = (1, 1)  # type 1 (good), type 2 (bad)
            n, d = sum(list(n_type_agents)) + sum(list(n_type_particles)), len(pind)  # everything
            # Custom state
            BHL = self.box_half_length
            state = np.zeros((n, d), dtype=np.float)  # format: (type -1, type -2, ..., type 1, type 2)
            state[idx,:] = np.array([-0.9*BHL, 0.0*BHL, 0.0, 0.0, -1]); idx+=1  # agent 1
            state[idx,:] = np.array([0.9*BHL, 0.0*BHL, 0.0, 0.0, -2]); idx+=1  # agent 2
            state[idx,:] = np.array([0.0*BHL, 0.9*BHL, 0.5, -0.5, 1]); idx+=1  # good particle
            state[idx,:] = np.array([0.0*BHL, -0.9*BHL, -0.5, 0.5, 2]); idx+=1  # bad particle
        else:
            idx = 0
            n_type_agents = (1, 0)  # type -1, type -2
            n_type_particles = (0, 110)  # type 1 (good), type 2 (bad)
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