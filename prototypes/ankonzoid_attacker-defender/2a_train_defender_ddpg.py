import numpy as np
import tensorflow as tf
import random
from collections import deque

if __name__ == "__main__":

    sess = tf.Session()

    from src.environment_cts import Environment_cts
    from src.agent_ddpg import Agent_DDPG
    env = Environment()
    agent = Agent_DDPG(env, sess)

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
