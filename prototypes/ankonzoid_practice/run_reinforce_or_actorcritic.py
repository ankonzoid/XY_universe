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
    env_name = "CartPole-v0"
else:
    env_name = "LunarLander-v2"

env = gym.make(env_name)
state_dim, num_actions = env.observation_space.shape[0], env.action_space.n
print("state_dim = {}, num_actions = {}".format(state_dim, num_actions))

# Tensorflow
sess = tf.Session()  # start session
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

# Agent
if 0:
    print("Using PG-REINFORCE agent!")
    from agent_reinforce import PolicyGradientREINFORCE
    agent = PolicyGradientREINFORCE(sess, state_dim, num_actions, summary_writer=writer)
else:
    print("Using PG-Actor-Critic agent!")
    from agent_actorcritic import PolicyGradientActorCritic
    agent = PolicyGradientActorCritic(sess, state_dim, num_actions, summary_writer=writer)

# Train
MAX_EPISODES, MAX_STEPS = 10000, 500
reward_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):

    state = env.reset()
    total_rewards = 0
    for iter in range(MAX_STEPS):
        #env.render()
        action = agent.get_action(state[np.newaxis,:])
        next_state, reward, done, _ = env.step(action)

        total_rewards += reward
        agent.store(state, action, reward)

        state = next_state
        if done: break

    agent.train()

    reward_history.append(total_rewards)
    mean_rewards = np.mean(reward_history)

    print("[episode {}/{}] iter = {}, rew = {}, rew_avg= {:.2f}".format(i_episode+1, MAX_EPISODES, iter+1, total_rewards, mean_rewards))
