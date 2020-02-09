"""

 XY_universe.py  (author: Anson Wong / git: ankonzoid)

 Trains a DQN agent to survive in the XY particle universe environment.

"""
import os, csv, datetime, random
import numpy as np
import pandas as pd
import tensorflow as tf
from src.Environment import Environment
from src.Agents import DQNAgent
from src.utils import mkdirs, save_plot, save_animation
tf.random.set_seed(1)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
random.seed(a=0)
np.random.seed(0)
output_dir = mkdirs(os.path.join(os.path.dirname(__file__), "output"))
models_dir = mkdirs(os.path.join(output_dir, "models"))
animations_dir = mkdirs(os.path.join(output_dir, "animations"))
log_file = os.path.join(output_dir, "log.csv")

env = Environment(n_good=0, n_bad=110)
agent = DQNAgent(env, n_sectors=4, sector_radius=1.0)
save_models = False
save_animations = True
n_episodes = 1000
iter_max = 2000
n_reward_max = 0
loss = -1 # track loss
for episode in range(n_episodes):
    iter = 0
    env.reset() # reset environment
    ob = agent.observe(env) # observe
    while iter < iter_max:
        action = agent.get_action(ob) # follow epsilon-greedy policy
        state_next, reward, done = env.step(action) # evolve
        ob_next = agent.observe(env) # observe
        agent.memorize((ob, action, reward, ob_next, done)) # save to replay buffer
        iter += 1
        if done:
            break # terminate
        ob = ob_next # transition
    # Save models/animations
    n_reward_max += (sum(env.reward_history) >= 2000) # track highly successful episodes
    print("[ep {}/{}] iter={}/{}, rew={:.0f}, nrewmax={}, mem={}, eps={:.3f}, loss={:.2f}".format(episode+1, n_episodes, iter, iter_max, sum(env.reward_history), n_reward_max, len(agent.memory), agent.epsilon, loss), flush=True)
    if (episode == 0 or n_reward_max % 5 == 1):
        if save_models:
            agent.save_model(os.path.join(models_dir, "model_ep={}_rew={}.h5".format(episode+1, int(sum(env.reward_history)))))
        if save_animations:
            save_animation(agent, env, filename=os.path.join(animations_dir, "xyuniverse_ep={}_rew={}.mp4".format(episode+1, int(sum(env.reward_history)))))
        n_reward_max += 0 if (episode == 0) else 1
    # Train agent
    loss = agent.train()
    # Save log
    header = ["episode", "iter", "reward", "loss", "epsilon", "time"]
    values = [episode + 1, iter, sum(env.reward_history), loss, agent.epsilon, datetime.datetime.now().strftime("%B %d %Y %I:%M:%S %p")]
    with open(log_file, ('w' if episode == 0 else 'a')) as f:
        writer = csv.writer(f)
        if episode == 0:
            writer.writerow(header)
        writer.writerow(values)
    # Plot log
    if (episode + 1) % 20 == 0 or (episode == n_episodes - 1):
        df = pd.read_csv(log_file)
        if df.shape[0] > 50:
            save_plot(xname="episode", yname="reward", df=df, color=(13/255, 28/255, 164/255), n_bins=50)
            save_plot(xname="episode", yname="loss", df=df, color=(195/255, 0/255, 0/255), n_bins=50)