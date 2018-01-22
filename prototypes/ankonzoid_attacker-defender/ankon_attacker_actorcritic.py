"""
solving pendulum using actor-critic model
"""

#import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque


# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
    def __init__(self, env, sess):

        self.name = "defender"

        #self.env = env
        self.observation_size = env.observation_size
        self.action_size = env.action_size
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=100000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        #self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])  # where we will feed de/dC (from critic)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=(self.observation_size,))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.action_size, activation='relu')(h3)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.observation_size,))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=(self.action_size,))
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            observation, action, reward, observation_new, _ = sample
            observation_reshaped = observation.reshape((1, self.observation_size))
            observation_new_reshaped = observation_new.reshape((1, self.observation_size))
            predicted_action = self.actor_model.predict(observation_reshaped)

            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: observation_reshaped,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: observation_reshaped,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            observation, action, reward, observation_new, done = sample
            observation_reshaped = observation.reshape((1, self.observation_size))
            observation_new_reshaped = observation_new.reshape((1, self.observation_size))
            action_reshaped = action.reshape((1, self.action_size))
            if not done:
                target_action = self.target_actor_model.predict(observation_new_reshaped)

                #print(target_action)
                future_reward = self.target_critic_model.predict(
                    [observation_new_reshaped, target_action])[0][0]
                reward += self.gamma * future_reward
            reward_reshaped = np.array([reward])
            self.critic_model.fit([observation_reshaped, action_reshaped], reward_reshaped, verbose=0)

    def train(self, n_batches=100, batch_size=32):
        if len(self.memory) < batch_size:
            return

        for session in range(n_batches):
            samples = random.sample(self.memory, batch_size)
            self._train_critic(samples)
            self._train_actor(samples)

    #def save(self, model_filename):
    #    self.critic_.save(model_filename)

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def get_action(self, observation):
        self.epsilon *= self.epsilon_decay
        observation_reshaped = observation.reshape((1, self.observation_size))
        if np.random.random() < self.epsilon:
            action = np.random.uniform(0, 1, self.action_size)
            action[0] = 5*action[0]
            action[1] = 2*np.pi*action[1]
            return action
        return self.actor_model.predict(observation_reshaped)[0]

"""
def __name__ == "__main__": 
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("Pendulum-v0")
    actor_critic = ActorCritic(env, sess)

    num_trials = 10000
    trial_len = 500

    cur_state = env.reset()
    action = env.action_space.sample()
    while True:
        #env.render()
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        action = actor_critic.act(cur_state)
        action = action.reshape((1, env.action_space.shape[0]))

        new_state, reward, done, _ = env.step(action)
        new_state = new_state.reshape((1, env.observation_space.shape[0]))

        actor_critic.remember(cur_state, action, reward, new_state, done)
        actor_critic.train()

        cur_state = new_state
"""

if __name__ == "__main__":

    sess = tf.Session()
    K.set_session(sess)

    from src.environment_cts import Environment
    env = Environment()
    agent = ActorCritic(env, sess)

    n_experiments = 1000
    n_episodes_per_experiment = 10
    n_iter = 300

    reward_avg_best = -9E9
    for experiment in range(n_experiments):

        reward_experiment = []
        n_iter_experiment = []
        ball_animation, defender_animation = [], []
        for episode in range(n_episodes_per_experiment):

            save_animation = (episode == 0)
            env.reset()  # reset experiment
            agent.memory = []  # reset agent memory

            if save_animation:
                ball_animation.append(env.ball.copy())  # store to animation data
                defender_animation.append(env.defender.copy())  # store to animation data

            iter, reward_episode = 0, 0.0
            while iter < n_iter:
                observation = env.make_observation()
                action = agent.get_action(observation)
                observation_next, reward, done = env.step(action)
                agent.remember(observation, action, reward, observation_next, done)

                if save_animation:
                    defender_animation.append(env.defender.copy())  # store
                    ball_animation.append(env.ball.copy())  # store
                reward_episode += reward  # sum up episodic reward
                iter += 1
                if done: break  # terminate

            reward_experiment.append(reward_episode)
            n_iter_experiment.append(iter)

        # Print experiment result
        reward_avg = sum(reward_experiment) / len(reward_experiment)
        print("experiment {}/{}: reward_avg = {:.3f} ({} episodes)".format(experiment, n_experiments, reward_avg, len(reward_experiment)))

        # Save model if the experiment has higher reward average
        if reward_avg > reward_avg_best:
            model_filename = "models/{}/model_exp={}_rewavg={:.3f}.h5".format(agent.name, experiment, reward_avg)
            animation_filename = "animations/{}/ani_ep={}_rew={:.3f}.mp4".format(agent.name, experiment, reward_avg)
            print(" -> saving better reward avg model '{}'".format(model_filename))
            reward_avg_best = reward_avg
            env.make_animation((defender_animation, ball_animation), animation_filename)
            #agent.save(model_filename)

        # Train model based on experiment data
        n_batches = n_episodes_per_experiment
        batch_size = int((sum(n_iter_experiment)/n_episodes_per_experiment)/10)
        agent.train(n_batches=n_batches, batch_size=batch_size)
