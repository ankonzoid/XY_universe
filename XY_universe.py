"""

 XY_universe.py  (author: Anson Wong / git: ankonzoid)

 Trains a DQN agent to survive in the XY particle universe environment.

"""
import os, csv, datetime, random
import numpy as np
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(1)
random.seed(a=0)
np.random.seed(0)
from src.Environment import Environment
from src.Sensors import Sensors
from src.Agents import DQNAgent
from src.utils import make_dir, save_plot

def main():
    # ====================
    # Run
    # ====================
    outDir = make_dir(os.path.join(os.path.dirname(__file__), "output"))
    modelDir = make_dir(os.path.join(outDir, "models"))
    aniDir = make_dir(os.path.join(outDir, "animations"))
    logFile = os.path.join(outDir, "log.csv")

    env = Environment(n_good=0, n_bad=110)
    agent = DQNAgent(env=env, sensors=Sensors(n_sectors=4, sector_radius=1.0))

    save_models = True
    save_animations = True
    n_episodes = 1000
    n_iter_max = 2000
    n_reward_max = 0
    loss = -1 # track training loss
    for episode in range(n_episodes):

        # Generate episode
        state = env.reset() # initialize environment
        obs = agent.observe(state, env) # observe
        states, rewards, i = [], [], 0
        while i < n_iter_max:
            action = agent.get_action(obs) # follow epsilon-greedy policy
            state_next, reward, done = env.step(state, action) # evolve
            obs_next = agent.observe(state, env) # observe
            agent.memorize((obs, action, reward, obs_next, done)) # memorize
            rewards.append(reward)
            states.append(state_next)
            state, obs = state_next, obs_next # transition
            i += 1
            if done: break # terminate

        # Track highly successful episodes
        n_reward_max += (sum(rewards) >= 2000)

        # Print progress
        print("[ep {}/{}] iter={}/{}, rew={:.0f}, nrewmax={}, mem={}, eps={:.3f}, loss={:.2f}".format(episode+1, n_episodes, i, n_iter_max, sum(rewards), n_reward_max, len(agent.memory), agent.epsilon, loss), flush=True)

        if (episode == 0 or n_reward_max % 5 == 1):
            # Save model
            if save_models:
                modelFile = os.path.join(modelDir, "model_ep={}_rew={}.h5".format(episode+1, int(sum(rewards))))
                print("    -> saving agent model = {}".format(modelFile), flush=True)
                agent.save_model(modelFile)
            # Save animation
            if save_animations:
                aniFile = os.path.join(aniDir, "ani_ep={}_rew={}.mp4".format(episode+1, int(sum(rewards))))
                print("    -> saving animation = {}".format(aniFile), flush=True)
                env.make_animation(states, env, agent.sensors, save_ani=True, filename=aniFile)
            n_reward_max += 0 if (episode == 0) else 1

        # Train agent
        loss = agent.train()  # automatically adjusts epsilon

        # Save log
        header = ["episode", "iter", "reward", "loss", "epsilon", "time"]
        values = [episode+1, i, sum(rewards), loss, agent.epsilon,
                  datetime.datetime.now().strftime("%B %d %Y %I:%M:%S %p")]
        with open(logFile, ('w' if episode == 0 else 'a')) as f:
            writer = csv.writer(f)
            if episode == 0:
                writer.writerow(header)
            writer.writerow(values)

        if (episode+1) % 20 == 0 or (episode == n_episodes-1):
            df = pd.read_csv(logFile)
            if df.shape[0] > 50:
                save_plot(["episode", "reward"], df, color=(13/255, 28/255, 164/255), n_bins=50)
                save_plot(["episode", "loss"], df, color=(195/255, 0/255, 0/255), n_bins=50)

# Driver
if __name__ == "__main__":
    main()