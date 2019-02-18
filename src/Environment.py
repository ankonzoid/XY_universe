"""

 Environment.py  (author: Anson Wong / git: ankonzoid)

 External functions:
   reset, make_observation, step, make_animation


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
class Environment():

    def __init__(self, universe, sensors):
        self.universe = universe
        self.sensors = sensors
        self.observation_size = sensors.observation_size
        self.action_size = universe.action_size

    def reset(self):
        state = self.universe.reset()
        return state

    def make_observation(self, state, pid_agent):
        observations = self.sensors.sense(state, pid_agent, self.universe)
        return observations

    def step(self, state, pid_agents, actions):
        state_next, rewards, dones = self.universe.step(state, pid_agents, actions)
        observations_next = []
        for pid_agent in pid_agents:
            observation_next = self.make_observation(state_next, pid_agent)
            observations_next.append(observation_next)
        return observations_next, state_next, rewards, dones

    def make_animation(self, states, save_animation=False, save_filename='animation.mp4'):
        self.universe.make_animation(states, self.universe, self.sensors,
                                     save_animation=save_animation, save_filename=save_filename)