"""

 DQNAgent.py  (author: Anson Wong / git: ankonzoid)

 Functions:
   get_action, train, store, save_model, load_model


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
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers

from AgentBase import AgentBase

class Agent(AgentBase):

    def __init__(self, settings, observation_size=None, action_size=None):

        # Exploration settings
        self.epsilon_range = settings.epsilon_range  # [max, min]
        self.epsilon_decay = settings.epsilon_decay  # multiply epsilon max by this after training
        self.epsilon = self.epsilon_range[0]
        self.gamma = settings.gamma  # discount parameter (important)

        # DQN architecture
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_layers = settings.hidden_layers
        self.L2_regularizer = settings.L2_regularizer
        self.activation = settings.activation

        # DQN training parameters
        self.learning_rate = settings.learning_rate  # value function learning rate
        self.loss = settings.loss  # "mse"
        self.n_epochs = settings.n_epochs  # number of training epochs
        self.verbose = False  # verbose about training?

        # For experience replay memory training
        self.n_training_sessions = settings.n_training_sessions  # training sessions per training episode
        self.training_batch_size = settings.training_batch_size  # training batch size
        self.memory_size = settings.memory_size  # # experiences stored to memory
        self.memory = deque(maxlen=self.memory_size)  # can pop oldest experiences

        # For target freezing model
        self.use_target_model = settings.use_target_model  # use and freeze target network?

        # Build DQN model
        self.model = self._build_model()  # actual model
        self.model_target = self._build_model()  # target model

    def _build_model(self):

        model = Sequential()
        for i, n_hidden_nodes in enumerate(self.hidden_layers):
            if i==0:
                model.add(Dense(self.hidden_layers[i], input_dim=self.observation_size, activation=self.activation,  kernel_regularizer=regularizers.l2(self.L2_regularizer[i])))
            else:
                model.add(Dense(self.hidden_layers[i], activation=self.activation,  kernel_regularizer=regularizers.l2(self.L2_regularizer[i])))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))

        return model

    # Get an action from our epsilon-greedy policy
    def get_action(self, observation):

        observation_reshaped = np.reshape(observation, [1, len(observation)])  # reshape for model input
        self.epsilon = max(self.epsilon_range)
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(list(range(self.action_size)))  # explore
        else:
            Q_estimate = self.model.predict(observation_reshaped)[0]
            action = np.random.choice(np.flatnonzero(Q_estimate==np.amax(Q_estimate)))  # exploit

        return action

    # Train our network using replay mini-batch experiences
    def train(self):
        loss_avg = 0.0
        for session in range(self.n_training_sessions):

            # Sample a mini-batch from replay memory
            minibatch = random.sample(self.memory, self.training_batch_size)
            (observations, actions, rewards, observations_next, dones) = ([], [], [], [], [])
            for (observation, action, reward, observation_next, done) in minibatch:
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                observations_next.append(observation_next)
                dones.append(done)

            observations = np.array(observations, dtype=np.float)  # (batch_size, observation_size)
            actions = np.array(actions, dtype=np.int)  # (batch_size, )
            rewards = np.array(rewards, dtype=np.float)  # (batch_size, )
            observations_next = np.array(observations_next, dtype=np.float)  # (batch_size, observation_size)
            dones = np.array(dones, dtype=np.int)  # (batch_size, )

            # Compute Q estimate
            Q = self.model.predict(observations)  # (batch_size, action_size)
            if self.use_target_model:
                Q_next = self.model_target.predict(observations_next)  # (batch_size, action_size)
            else:
                Q_next = self.model.predict(observations_next)  # (batch_size, action_size)

            # Compute target Q
            Q_target = Q.copy()  # (batch_size, action_size)
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                Q_target[i, action] = reward
                if not done:
                    Q_target[i, action] += self.gamma * np.amax(Q_next[i,:])
                loss_avg += np.abs(Q[i, action] - Q_target[i, action])**2

            # Train model
            self.model.fit(observations, Q_target, epochs=self.n_epochs, verbose=self.verbose)

        # Compute average loss
        loss_avg /= self.training_batch_size
        loss_avg /= self.n_training_sessions

        # Decay agent epsilon max value
        self.epsilon_range[0] *= self.epsilon_decay

        # Update target network
        if self.use_target_model:
            self._update_target_model()

        return loss_avg

    # Store to agent memory where experience = (observation, action, reward, observation_next, done)
    def store(self, experience):
        self.memory.append(experience)

    # Save agent Q-model
    def save_model(self, filename):
        self.model.save(filename)

    # Load Q-model into agent
    def load_model(self, filename):
        from keras.models import load_model
        self.model = load_model(filename)
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))

    # ============================================
    # Secondary functions
    # ============================================

    # Update target network
    def _update_target_model(self):
        self.model_target.set_weights(self.model.get_weights())  # transfer weights to target model