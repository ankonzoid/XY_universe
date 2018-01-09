from agent import Agent
from environment import Environment

if __name__ == "__main__":

    model_filename_load = "models/model_exp=170_rewavg=0.940.h5"

    env = Environment()
    agent = Agent(observation_size=env.observation_size, action_size=env.action_size)
    agent.load(model_filename_load)  # load trained model

    n_episodes_per_experiment = 1000
    n_iter = 300
    n_animations_save = 10

    reward_avg_best = -9E9
    for experiment in range(1):

        reward_experiment = []
        n_iter_experiment = []
        for episode in range(n_episodes_per_experiment):

            env.reset()  # reset experiment
            agent.memory = []  # reset agent memory
            ball_animation, defender_animation = [], []  # reset animation data

            ball_animation.append(env.ball.copy())  # store to animation data
            defender_animation.append(env.defender.copy())  # store to animation data

            reward_episode = 0.0
            iter = 0
            while iter < n_iter:
                observation = env.make_observation()
                action = agent.get_action(observation)
                observation_next, reward, done = env.step(action)
                agent.memory.append([observation, action, reward, observation_next, done])

                defender_animation.append(env.defender.copy())  # store
                ball_animation.append(env.ball.copy())  # store
                reward_episode += reward  # sum up episodic reward
                iter += 1
                if done: break  # terminate

            reward_experiment.append(reward_episode)
            n_iter_experiment.append(iter)

            print("[episode {}/{}] reward={:.3f}".format(episode+1, n_episodes_per_experiment, reward_episode))

            if episode < n_animations_save:
                animation_data = (defender_animation, ball_animation)
                env.make_animation(animation_data, "animations/ani_ep={}_rew={:.3f}.mp4".format(episode, reward_episode))

        # Print experiment result
        reward_avg = sum(reward_experiment) / len(reward_experiment)
        print("Evaluation: reward_avg = {:.3f} ({} episodes)".format(reward_avg, len(reward_experiment)))