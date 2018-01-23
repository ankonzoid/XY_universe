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
MAX_EPISODES, MAX_STEPS = 10000, 200
reward_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):
    state = env.reset()
    total_rewards = 0

    for t in range(MAX_STEPS):
        #env.render()
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
