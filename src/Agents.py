"""

 Agents.py  (author: Anson Wong / git: ankonzoid)


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
import numpy as np
import random
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers

class DQNAgent():

    def __init__(self, env, sensors):

        # DQN agent settings
        self.observation_size = sensors.observation_size
        self.action_size = env.action_size
        self.sensors = sensors
        self.epsilon_range = [0.2, 0.01] # [max, min]
        self.epsilon_decay = 0.995 # multiply epsilon max by this after training
        self.epsilon = self.epsilon_range[0] # starting exploration parameter
        self.gamma = 0.99  # discount parameter (important)
        self.hidden_layers = [100, 100]
        self.L2_regularizer = [1E-6] * len(self.hidden_layers)
        self.learning_rate = 0.005 # value function learning rate
        self.activation = "relu"
        self.loss = "mse"
        self.n_epochs = 1 # epochs per fit() call
        self.memory_size = 50000 # max number of experiences stored in memory
        self.batch_size = 200 # training batch size
        self.n_training_sessions = 50 # sessions per train() call
        self.verbose = False

        # For experience replay memory training
        self.memory = deque(maxlen=self.memory_size)  # can pop oldest experiences

        # Build DQN model
        self.model = self._build_model()  # actual model

    # Make observation of state
    def observe(self, state, env):
        obs, _ = self.sensors.sense(state, 0, env)
        return obs

    # Follow epsilon-greedy policy
    def get_action(self, obs):
        obs_reshaped = np.reshape(obs, [1, len(obs)])
        self.epsilon = max(self.epsilon_range)
        rand = random.uniform(0, 1)
        if rand <= self.epsilon: # explore
            action = random.choice(list(range(self.action_size)))
        else: # exploit
            Qpred = self.model.predict(obs_reshaped)[0]
            action = random.choice(np.flatnonzero(Qpred==np.amax(Qpred)))
        return action

    # Train our network using replay mini-batch experiences
    def train(self):

        # Check there is enough memory for training
        if len(self.memory) < self.batch_size:
            return -1.0

        # Train
        loss_avg = 0.0
        for session in range(self.n_training_sessions):

            # Sample a mini-batch from replay memory
            minibatch = random.sample(self.memory, self.batch_size)
            obs_arr, action_arr, reward_arr, obs_next_arr, done_arr = [], [], [], [], []
            for (obs, action, reward, obs_next, done) in minibatch:
                obs_arr.append(obs)
                action_arr.append(action)
                reward_arr.append(reward)
                obs_next_arr.append(obs_next)
                done_arr.append(done)
            obs_arr = np.array(obs_arr, dtype=np.float)
            action_arr = np.array(action_arr, dtype=np.int)
            reward_arr = np.array(reward_arr, dtype=np.float)
            obs_next_arr = np.array(obs_next_arr, dtype=np.float)
            done_arr = np.array(done_arr, dtype=np.int)

            # Predict Q and Q_next
            Q = self.model.predict(obs_arr)  # (batch_size, action_size)
            Q_next = self.model.predict(obs_next_arr)  # (batch_size, action_size)

            # Compute target Q
            Q_target = Q.copy()  # (batch_size, action_size)
            for i, (action, reward, done) in enumerate(zip(action_arr, reward_arr, done_arr)):
                Q_target[i, action] = reward
                if not done:
                    Q_target[i, action] += self.gamma * np.amax(Q_next[i,:])
                loss_avg += np.abs(Q[i, action] - Q_target[i, action])**2

            # Train model
            self.model.fit(obs_arr, Q_target, epochs=self.n_epochs, verbose=self.verbose)

        # Compute average loss
        loss_avg /= self.batch_size
        loss_avg /= self.n_training_sessions

        # Decay agent epsilon max value
        self.epsilon_range[0] *= self.epsilon_decay

        return loss_avg

    # Store to agent memory where experience = (observation, action, reward, observation_next, done)
    def memorize(self, experience):
        self.memory.append(experience)

    # Save agent Q-model
    def save_model(self, filename):
        self.model.save(filename)

    # Load Q-model into agent
    def load_model(self, filename):
        self.model = load_model(filename)
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))

    def write_settings(self, filename):
        f = open(filename, 'w')
        f.write('\n')
        attrs = vars(self)
        for item in sorted(attrs.items()):
            (item_name, item_value) = item
            if item_name not in ["memory", "model", "model_target"]:
                f.write('agent.{} = {}\n'.format(item_name, item_value))

    # ============================================
    # Secondary functions
    # ============================================

    # Build FC network
    def _build_model(self):
        model = Sequential()
        for i, n_hidden_nodes in enumerate(self.hidden_layers):
            if i==0:
                model.add(Dense(self.hidden_layers[i],
                                input_dim=self.observation_size,
                                activation=self.activation,
                                kernel_regularizer=regularizers.l2(self.L2_regularizer[i])))
            else:
                model.add(Dense(self.hidden_layers[i],
                                activation=self.activation,
                                kernel_regularizer=regularizers.l2(self.L2_regularizer[i])))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))
        return model