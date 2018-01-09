import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

class Agent:

    def __init__(self, name="agent", observation_size=2, action_size=2):
        self.name = name
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 0.1

        self.loss = "mse"
        self.learning_rate = 0.001

        self._build_model()
        self._reserve_memory()

    def _reserve_memory(self):
        self.memory = []

    def _build_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.observation_size, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))
        self.model = model

    def train(self, n_batches=10, batch_size=200):

        print(" -> training with n_batches={} and batch_size={}".format(n_batches, batch_size))
        for session in range(n_batches):

            minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
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
            Q_next = self.model.predict(observations_next)  # (batch_size, action_size)

            # Compute target Q
            Q_target = Q.copy()  # (batch_size, action_size)
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                Q_target[i, action] = reward
                if not done:
                    Q_target[i, action] += self.gamma * np.amax(Q_next[i, :])

            # Train model
            self.model.fit(observations, Q_target, epochs=1, verbose=False)

    def get_action(self, observation):

        observation_reshaped = np.reshape(observation, [1, len(observation)])  # reshape for model input
        if np.random.rand() < self.epsilon:
            action = np.random.choice(list(range(self.action_size)))  # explore
        else:
            Q_estimate = self.model.predict(observation_reshaped)[0]
            action = np.random.choice(np.flatnonzero(Q_estimate == np.amax(Q_estimate)))  # exploit

        return action

    def save(self, model_filename):
        self.model.save(model_filename)

    def load(self, model_filename):
        self.model = load_model(model_filename)
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))
