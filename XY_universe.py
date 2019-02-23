"""

 XY_universe.py  (author: Anson Wong / git: ankonzoid)

 Trains a DQN agent to survive in the XY particle universe environment.


 Copyright 2019 Anson Wong

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

"""
import os, csv, datetime, random
import numpy as np
import tensorflow as tf
from src.Environment import Environment # agent environment based on sensors
from src.Sensors import Sensors # sensors for universe
from src.Agents import DQNAgent
tf.set_random_seed(2)
random.seed(a=0)
np.random.seed(0)

def main():
    # ====================
    # Settings
    # ====================
    save = True
    outDir = os.path.join(os.path.dirname(__file__), "output")
    modelDir = os.path.join(outDir, "models")
    aniDir = os.path.join(outDir, "animations")
    if not os.path.exists(outDir): os.makedirs(outDir)
    if not os.path.exists(modelDir): os.makedirs(modelDir)
    if not os.path.exists(aniDir): os.makedirs(aniDir)

    env = Environment(n_good=0, n_bad=110)
    agent = DQNAgent(env=env, sensors=Sensors(n_sectors=4, sector_radius=1.0))

    # ====================
    # Run
    # ====================
    n_episodes = 500
    n_iter = 2000
    n_reward_save = 0
    loss = -1 # track training loss
    for episode in range(n_episodes):

        # Generate episode following epsilon-greedy policy
        state = env.reset() # initialize universe
        obs = agent.observe(state, env) # observe universe
        states, rewards, i = [], [], 0
        while i < n_iter:
            action = agent.get_action(obs) # follow policy
            state_next, reward, done = env.step(state, action) # evolve
            obs_next = agent.observe(state, env) # observe
            agent.memorize((obs, action, reward, obs_next, done)) # memorize
            rewards.append(reward)
            states.append(state_next)
            state, obs = state_next, obs_next # transition
            i += 1
            if done: break # terminate

        # Track highly successful episodes
        n_reward_save += (sum(rewards) >= 2000)

        # Print and check if we hit reward threshold
        print("[ep {}/{}] iter={}/{}, rew={:.0f}, mem={}, eps={:.5f}, loss={:.2f}".format(episode+1, n_episodes, i, n_iter, sum(rewards), len(agent.memory), agent.epsilon, loss), flush=True)

        if save and (episode == 0 or n_reward_save % 10 == 1):
            # Save model
            modelFile = os.path.join(modelDir, "model_ep={}_rew={}.h5".format(episode+1, int(sum(rewards))))
            print("    -> saving agent model = {}".format(modelFile), flush=True)
            agent.save_model(modelFile)
            # Save animation
            aniFile = os.path.join(aniDir, "ani_ep={}_rew={}.mp4".format(episode+1, int(sum(rewards))))
            print("    -> saving animation = {}".format(aniFile), flush=True)
            env.make_animation(states, env, agent.sensors, save_ani=True, filename=aniFile)
            n_reward_save += 0 if (episode == 0) else 1

        # Train agent
        loss = agent.train()  # automatically adjusts epsilon

        # Save log
        if save:
            header = ["episode", "iter", "reward", "loss", "epsilon", "time"]
            values = [episode+1, i, sum(rewards), loss, agent.epsilon,
                      datetime.datetime.now().strftime("%B %d %Y %I:%M:%S %p")]
            logFile = os.path.join(outDir, "log.csv")
            with open(logFile, ('w' if episode == 0 else 'a')) as f:
                writer = csv.writer(f)
                if episode == 0:
                    writer.writerow(header)
                writer.writerow(values)

# Driver
if __name__ == "__main__":
    main()