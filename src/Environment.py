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

    def __init__(self, universe):
        self.universe = universe

    def reset(self):
        state = self.universe.reset()
        return state

    def step(self, state, actions):
        state_next, reward, done = self.universe.step(state, actions)
        return state_next, reward, done

    def make_animation(self, states, save_ani=False, filename='animation.mp4'):
        self.universe.make_animation(states, self.universe, self.sensors,
                                     save_ani=save_ani, filename=filename)

