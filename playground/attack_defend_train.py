from agent import Agent
from environment import Environment

if __name__ == "__main__":

    env = Environment()
    agent = Agent(observation_size=env.observation_size, action_size=env.action_size)
    n_experiments = 1000
    n_episodes_per_experiment = 100
    n_iter = 300

    reward_avg_best = -9E9
    for experiment in range(n_experiments):

        reward_experiment = []
        n_iter_experiment = []
        for episode in range(n_episodes_per_experiment):

            env.reset()  # reset experiment
            agent.memory = []  # reset agent memory

            reward_episode = 0.0
            iter = 0
            while iter < n_iter:
                observation = env.make_observation()
                action = agent.get_action(observation)
                observation_next, reward, done = env.step(action)
                agent.memory.append([observation, action, reward, observation_next, done])

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
            model_filename = "models/model_exp={}_rewavg={:.3f}.h5".format(experiment, reward_avg)
            print(" -> saving better reward avg model '{}'".format(model_filename))
            reward_avg_best = reward_avg
            agent.save(model_filename)

        # Train model based on experiment data
        n_batches = n_episodes_per_experiment
        batch_size = int((sum(n_iter_experiment)/n_episodes_per_experiment)/10)
        agent.train(n_batches=n_batches, batch_size=batch_size)
