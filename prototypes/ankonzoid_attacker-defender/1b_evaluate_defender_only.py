from src.agent_dqn import Agent_DQN
from src.environment_discrete import Environment_discrete

if __name__ == "__main__":

    name = "defender"
    save_animations = False
    model_filename_load = "models/defender/model_exp=194_rewavg=0.960.h5"

    env = Environment_discrete()
    agent = Agent_DQN(name=name,
                      observation_size=env.observation_size,
                      action_size=env.action_size)
    agent.load(model_filename_load)  # load trained model

    n_episodes_per_experiment = 1000
    n_iter = 300
    n_animations_save = 10

    for experiment in range(1):

        reward_experiment = []
        n_iter_experiment = []
        for episode in range(n_episodes_per_experiment):

            env.reset()  # reset experiment
            agent.memory = []  # reset agent memory
            ball_animation, defender_animation = [], []  # reset animation data

            ball_animation.append(env.ball.copy())  # store to animation data
            defender_animation.append(env.defender.copy())  # store to animation data

            iter, reward_episode = 0, 0.0
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

            if episode < n_animations_save and save_animations:
                animation_data = (defender_animation, ball_animation)
                env.make_animation(animation_data, "animations/{}/ani_ep={}_rew={:.3f}.mp4".format(name, episode, reward_episode))

        # Print experiment result
        reward_avg = sum(reward_experiment) / len(reward_experiment)
        print("-> reward_avg = {:.3f} (over {} episodes)".format(reward_avg, len(reward_experiment)))