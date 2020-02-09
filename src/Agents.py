"""

 Agents.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from src.Sensors import Sensors

class DQNAgent():

    def __init__(self, env, n_sectors, sector_radius):
        self.sensors = Sensors(n_sectors, sector_radius)
        self.action_size = env.action_size
        self.epsilon_range = [0.20, 0.20] # [max, min]
        self.epsilon_decay = 0.99 # multiply epsilon max by this after training
        self.epsilon = self.epsilon_range[0] # starting exploration parameter
        self.gamma = 0.99  # discount parameter (important)
        self.n_training_sessions = 50 # training sessions per train() call
        self.batch_size = 200 # training batch size
        self.memory = deque(maxlen=20000) # replay buffer
        self.model = Sequential() # DQN
        self.model.add(Dense(60, input_dim=self.sensors.observation_size, activation="relu"))
        self.model.add(Dense(60, activation="relu"))
        self.model.add(Dense(self.action_size, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=1E-3))

    def observe(self, env):
        ob, sector_observation, sectors = self.sensors.sense(env)
        return ob

    def get_action(self, ob):
        self.epsilon = max(self.epsilon_range)
        if random.uniform(0, 1) <= self.epsilon: # explore
            action = random.choice(list(range(self.action_size)))
        else: # exploit
            Qpred = self.model.predict(np.reshape(ob, [1, len(ob)]))[0]
            action = random.choice(np.flatnonzero(Qpred == np.amax(Qpred)))
        return action

    def train(self):
        if len(self.memory) < self.batch_size:
            return -1.0
        loss_avg = 0.0
        for session in range(self.n_training_sessions): # train DQN on mini-batches of replay buffer
            minibatch = random.sample(self.memory, self.batch_size) # sample replay buffer
            obs, actions, rewards, obs_next, dones = [], [], [], [], []
            for (ob, action, reward, ob_next, done) in minibatch:
                obs.append(ob)
                actions.append(action)
                rewards.append(reward)
                obs_next.append(ob_next)
                dones.append(done)
            obs = np.array(obs, dtype=np.float)
            actions = np.array(actions, dtype=np.int)
            rewards = np.array(rewards, dtype=np.float)
            obs_next = np.array(obs_next, dtype=np.float)
            dones = np.array(dones, dtype=np.int)
            Q = self.model.predict(obs) # current Q[obs,:] estimate
            Q_next = self.model.predict(obs_next) # current Q[obs_next,:] estimate
            Q_target = Q.copy() # construct target Q
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                Q_target[i, action] = (reward + self.gamma * np.amax(Q_next[i,:])) if not done else reward
                loss_avg += np.abs(Q[i, action] - Q_target[i, action])**2
            self.model.fit(obs, Q_target, epochs=1, verbose=False) # train
        loss_avg /= self.batch_size
        loss_avg /= self.n_training_sessions
        self.epsilon_range[0] *= self.epsilon_decay # decary agent exploration
        tf.keras.backend.clear_session() # temporary fix for memory leak in tf 2.0
        return loss_avg

    def memorize(self, memory):
        self.memory.append(memory) # store memory = (ob, action, reward, ob_next, done)

    def save_model(self, filename):
        print("    -> saving agent model = {}".format(filename), flush=True)
        self.model.save(filename) # save agent Q

    def load_model(self, filename):
        self.model = load_model(filename) # load agent Q
        self.model.compile(loss="mse", optimizer=Adam(lr=1E-3))