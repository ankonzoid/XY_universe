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
    env_name = "LunarLanderContinuous-v2"

env = gym.make(env_name)
state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
action_high, action_low = env.action_space.high, env.action_space.low
print("state_dim = {}, action_dim = {}".format(state_dim, action_dim))
print("action_high = {}, action_low = {}".format(action_high, action_low))

# Tensorflow
sess = tf.Session()  # start session
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

# Agent
if 1:
    print("Using DDPG agent!")
    from agent_ddpg import DeepDeterministicPolicyGradient
    agent = DeepDeterministicPolicyGradient(sess, state_dim, action_dim,
                                            action_low, action_high, summary_writer=writer)

# Train
MAX_EPISODES, MAX_STEPS = 10000, 5000
total_rewards_history = deque(maxlen=50)
for i_episode in range(MAX_EPISODES):

    state = env.reset()
    total_rewards_training = 0
    total_iter_training = 0
    for iter in range(MAX_STEPS):
        #env.render()
        action = agent.get_action(state[np.newaxis,:])
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)  # store into replay buffer
        agent.train()  # train agent
        total_iter_training += 1
        total_rewards_training += reward
        state = next_state
        if done: break
    total_rewards_history.append(total_rewards_training)

    print("[episode {}/{}] iter = {}, rew = {:.2f}, rew avg = {:.2f}...".format(i_episode+1, MAX_EPISODES, total_iter_training, total_rewards_training, np.mean(total_rewards_history)))

    # For every 500 episodes, we evaluate our agent greedily (no training involved)
    #if i_episode % 500 == 0:
    if 0:
        n_evaluation_episodes = 100
        print("  -> Evaluating agent model greedily using {} episodes...".format(n_evaluation_episodes))
        experiment_iter = []
        experiment_rewards = []
        for experiment in range(n_evaluation_episodes):
            total_rewards = 0
            state = env.reset()
            for iter in range(MAX_STEPS):
                #env.render()
                action = agent.get_action(state[np.newaxis, :], exploration=False)  # turn off exploration
                next_state, reward, done, _ = env.step(action)
                total_rewards += reward  # collect total reward
                state = next_state
                if done: break
            experiment_iter.append(iter)
            experiment_rewards.append(total_rewards)
        print("  -> iter_avg = {:.2f}, rew_avg = {:.2f}".format(np.mean(experiment_iter), np.mean(experiment_rewards)))