"""

 run.py  (author: Anson Wong / git: ankonzoid)

"""
from __future__ import print_function
from collections import deque

import tensorflow as tf
import numpy as np
import gym

# Gym environment
env_name = "CartPole-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
print("state_dim = {}, num_actions = {}".format(state_dim, num_actions))

# Tensorflow
sess = tf.Session()  # start session
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)  # set optimizer
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

# Policy network with 1 hidden layer
def policy_network(x_input):
    n_hidden_1 = 20
    W1 = tf.get_variable("W1", [state_dim, n_hidden_1], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [n_hidden_1], initializer=tf.constant_initializer(0))
    h1 = tf.nn.tanh(tf.matmul(x_input, W1) + b1) # first layer: tanh(s*W1+b1)

    W2 = tf.get_variable("W2", [n_hidden_1, num_actions], initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", [num_actions], initializer=tf.constant_initializer(0))
    x_output = tf.matmul(h1, W2) + b2  # second layer: s*W2+b2
    return x_output

from agent_reinforce import PolicyGradientREINFORCE
agent = PolicyGradientREINFORCE(sess,
                                optimizer,
                                policy_network,
                                state_dim, num_actions,
                                summary_writer=writer)

MAX_EPISODES = 10000
MAX_STEPS = 200

reward_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):
    state = env.reset()
    total_rewards = 0

    for t in range(MAX_STEPS):
        env.render()
        action = agent.get_action(state[np.newaxis,:])
        next_state, reward, done, _ = env.step(action)

        total_rewards += reward
        reward = -10 if done else 0.1 # normalize reward
        agent.store(state, action, reward)

        state = next_state
        if done: break

    agent.train()

    reward_history.append(total_rewards)
    mean_rewards = np.mean(reward_history)

    print("Episode {}".format(i_episode))
    print("Finished after {} timesteps".format(t+1))
    print("Reward for this episode: {}".format(total_rewards))
    print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
    if mean_rewards >= 195.0 and len(reward_history) >= 100:
        print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
        break
