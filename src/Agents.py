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
        self.epsilon_range = [0.2, 0.2] # [max, min]
        self.epsilon_decay = 1.0 # multiply epsilon max by this after training
        self.epsilon = self.epsilon_range[0] # starting exploration parameter
        self.gamma = 0.99  # discount parameter (important)
        self.hidden_layers = [100, 100]
        self.L2_regularizer = [1E-5, 1E-5]
        self.learning_rate = 0.005 # value function learning rate
        self.activation = "relu"
        self.loss = "mse"
        self.n_epochs = 1 # number of training epochs
        self.memory_size = 10000 # experiences stored to memory
        self.training_batch_size = 200 # training batch size
        self.n_training_sessions = 20 # training sessions per training episode
        self.verbose = False

        # For experience replay memory training
        self.memory = deque(maxlen=self.memory_size)  # can pop oldest experiences

        # Build DQN model
        self.model = self._build_model()  # actual model

    # Make observation of state
    def observe(self, state, env):
        obs = self.sensors.sense(state, env)
        return obs

    # Follow epsilon-greedy policy
    def get_action(self, obs):
        obs_reshaped = np.reshape(obs, [1, len(obs)])
        self.epsilon = max(self.epsilon_range)
        if np.random.rand() <= self.epsilon: # explore
            action = np.random.choice(list(range(self.action_size)))
        else: # exploit
            Qpred = self.model.predict(obs_reshaped)[0]
            action = np.random.choice(np.flatnonzero(Qpred==np.amax(Qpred)))
        return action

    # Train our network using replay mini-batch experiences
    def train(self):
        loss_avg = 0.0
        for session in range(self.n_training_sessions):

            # Sample a mini-batch from replay memory
            minibatch = random.sample(self.memory, min(self.training_batch_size, len(self.memory)))
            (obs, actions, rewards, obs_next, dones) = ([], [], [], [], [])
            for (observation, action, reward, obs_next, done) in minibatch:
                obs.append(observation)
                actions.append(action)
                rewards.append(reward)
                obs_next.append(obs_next)
                dones.append(done)

            obs = np.array(obs, dtype=np.float)  # (batch_size, observation_size)
            actions = np.array(actions, dtype=np.int)  # (batch_size, )
            rewards = np.array(rewards, dtype=np.float)  # (batch_size, )
            obs_next = np.array(obs_next, dtype=np.float)  # (batch_size, observation_size)
            dones = np.array(dones, dtype=np.int)  # (batch_size, )

            # Compute Q estimate
            Q = self.model.predict(obs)  # (batch_size, action_size)
            if self.use_target_model:
                Q_next = self.model_target.predict(obs_next)  # (batch_size, action_size)
            else:
                Q_next = self.model.predict(obs_next)  # (batch_size, action_size)

            # Compute target Q
            Q_target = Q.copy()  # (batch_size, action_size)
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                Q_target[i, action] = reward
                if not done:
                    Q_target[i, action] += self.gamma * np.amax(Q_next[i,:])
                loss_avg += np.abs(Q[i, action] - Q_target[i, action])**2

            # Train model
            self.model.fit(obs, Q_target, epochs=self.n_epochs, verbose=self.verbose)

        # Compute average loss
        loss_avg /= self.training_batch_size
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