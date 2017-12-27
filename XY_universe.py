"""

 XY_universe.py  (author: Anson Wong / git: ankonzoid)

 Trains an agent play in the XY universe environment.


 Copyright 2017 Anson Wong

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
import sys, os
root_dir = os.path.join(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "src"))
sys.path.append(os.path.join(root_dir, "src", "bases"))
from Universe import Universe
from Sensors import Sensors
from Environment import Environment
from Config import Config

from agents.DQNAgent import Agent
from settings.bump_1 import Settings

def main():
    # ====================
    # Settings
    # ====================
    settings = Settings()
    universe, sensors = Universe(settings), Sensors(settings)
    env = Environment(universe, sensors)
    pid_agents, n_agents = env.universe.pid_agents, env.universe.n_agents
    agents = [Agent(settings, observation_size=env.observation_size, action_size=env.action_size) for i in range(n_agents)]
    cfg = Config(env, root_dir, settings)
    if settings.save_settings:
        cfg.write_settings(universe, sensors, agents[0])
    n_episodes, n_iter = settings.n_episodes, settings.n_iter

    # ====================
    # Run
    # ====================
    for episode in range(n_episodes):

        # Generate episode
        states_episode, iter_episode, rewards_episode, losses_episode = [], 0, [0]*n_agents, [-1]*n_agents
        state = env.reset()
        obs = [env.make_observation(state, pid_agent) for pid_agent in pid_agents]
        for iter in range(n_iter):
            actions = [agents[i].get_action(obs[i]) for i in range(n_agents)]  # get action
            obs_next, state_next, rewards, dones = env.step(state, pid_agents, actions)  # evolve
            for i in range(n_agents):
                agents[i].store((obs[i], actions[i], rewards[i], obs_next[i], dones[i]))  # append
                rewards_episode[i] += rewards[i]
            iter_episode += 1
            states_episode.append(state_next)
            state, observations = state_next, obs_next  # transition
            if True in dones:  # termination condition
                break

        # Print and check if we hit reward threshold
        print("[ep {:d}/{:d}] iter: {:d}/{:d}".format(episode+1, n_episodes, iter_episode, n_iter), flush=True)
        condition_save = False
        for i in range(n_agents):
            print("  (agent-{:d}) eps: {:.3f}, rew: {:.2f}, mem: {:d}".format(i, agents[i].epsilon, rewards_episode[i], len(agents[i].memory)), end='\n', flush=True)
            cfg.n_reward_threshold_save[i] += (rewards_episode[i] >= cfg.reward_threshold_save)
            if (episode == 0) or (cfg.n_reward_threshold_save[i] % cfg.episodes_reward_threshold_save == 1):
                condition_save = True
                cfg.n_reward_threshold_save[i] += (rewards_episode[i] >= cfg.reward_threshold_save)

        # Save models + animations
        if condition_save:
            if cfg.save_model:
                for i in range(n_agents):
                    print("    -> saving agent-{} model: '{}'".format(i, cfg.model_filename(episode, i)), flush=True)
                    agents[i].save_model(cfg.model_filename(episode, i))
            if cfg.save_animation:
                print("    -> saving animation: '{}'".format(cfg.animation_filename(episode)), flush=True)
                env.make_animation(states_episode, save_animation=True, save_filename=cfg.animation_filename(episode))

        # Train agents
        for i in range(n_agents):
            if len(agents[i].memory) >= agents[i].training_batch_size:
                losses_episode[i] = agents[i].train()  # automatically adjusts epsilon
                print("  -> training agent-{}... sessions={}, freeze={}, batch: {:d}, epochs: {:d}, loss: {:.6f}".format(i, agents[i].n_training_sessions, agents[i].use_target_model, agents[i].training_batch_size, agents[i].n_epochs, losses_episode[i]), flush=True)

        # Save log
        if cfg.save_log:
            headers, values = ["episode", "iter"], [episode, iter_episode]
            for i in range(n_agents):
                headers.extend(["rew_{}".format(i), "loss_{}".format(i), "eps_{}".format(i)])
                values.extend([rewards_episode[i], losses_episode[i], agents[i].epsilon])
            cfg.write_log(headers, values)

# Driver
if __name__ == "__main__":
    main()