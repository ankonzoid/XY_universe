"""

 run.py  (author: Anson Wong / git: ankonzoid)

"""
from __future__ import print_function
from collections import deque
import tensorflow as tf
import numpy as np
import gym

# Gym environment
if 1:
    env_name = "InvertedPendulum-v1"

env = gym.make(env_name)
state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
print("state_dim = {}, action_dim = {}".format(state_dim, action_dim))

# Tensorflow
sess = tf.Session()  # start session
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

# Agent
if 1:
    print("Using DDPG agent!")
    from agent_ddpg import DeepDeterministicPolicyGradient
    agent = DeepDeterministicPolicyGradient(sess, state_dim, action_dim, summary_writer=writer)

# Train
MAX_EPISODES, MAX_STEPS = 10000, 1000
episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):
    state = env.reset()
    for t in range(MAX_STEPS):
        #env.render()
        action = agent.get_action(state[np.newaxis,:])
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()  # training is done in the loop
        state = next_state
        if done: break

    if i_episode % 500 == 0:

        for i_eval in range(100):
            total_rewards = 0
            state = env.reset()
            for t in range(MAX_STEPS):
                # env.render()
                action = agent.get_action(state[np.newaxis, :], exploration=False)
                next_state, reward, done, _ = env.step(action)
                total_rewards += reward
                state = next_state
                if done: break

            episode_history.append(total_rewards)
            mean_rewards = np.mean(episode_history)

        print("Episode {}".format(i_episode))
        print("Finished after {} timesteps".format(t + 1))
        print("Reward for this episode: {:.2f}".format(total_rewards))
        print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
        # if mean_rewards >= 950.0: # for InvertedPendulum-v1
        if mean_rewards >= 9100.0:  # for InvertedDoublePendulum-v1
            print("Environment {} solved after {} episodes".format(env_name, i_episode + 1))
            break
