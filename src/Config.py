"""

 Config.py  (author: Anson Wong / git: ankonzoid)

 External functions:
   make_dir, model_filename, animation_filename, write_settings, write_log


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
import os, csv, datetime

class Config():

    def __init__(self, env, root_dir, settings):

        # Project output directory
        self.output_dir = os.path.join(root_dir, "output", env.universe.name)
        self._make_dir(self.output_dir)

        # Settings
        self.save_settings = settings.save_settings
        self.settings_filename = os.path.join(self.output_dir, "settings.txt".format(env.universe.name))

        # Log
        self.save_log = settings.save_log
        self.n_log = 0
        self.log_filename = os.path.join(self.output_dir, "log.csv".format(env.universe.name))

        # Models
        self.save_model = settings.save_model
        self.models_dir = os.path.join(self.output_dir, "models")
        self._make_dir(self.models_dir)

        # Animations
        self.save_animation = settings.save_animation
        self.animations_dir = os.path.join(self.output_dir, "animations")
        self._make_dir(self.animations_dir)

        # Reward threshold for saving (models/animations)
        self.reward_threshold_save = settings.reward_threshold_save  # reward threshold for logging
        self.episodes_reward_threshold_save = settings.episodes_reward_threshold_save
        self.n_reward_threshold_save = [0] * len(settings.pid_agents)

    def _make_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def model_filename(self, episode, i_agent):
        return os.path.join(self.models_dir, "model_ep{}_ag{}.h5".format(episode, i_agent))

    def animation_filename(self, episode):
        return os.path.join(self.animations_dir, "anim_ep{}.mp4".format(episode))

    def write_settings(self, universe, sensors, agent):
        if self.save_settings:
            f = open(self.settings_filename, 'w')

            attrs = vars(universe)
            for item in sorted(attrs.items()):
                (item_name, item_value) = item
                if item_name not in []:
                    f.write('universe.{} = {}\n'.format(item_name, item_value))

            f.write('\n')
            attrs = vars(sensors)
            for item in sorted(attrs.items()):
                (item_name, item_value) = item
                if item_name not in []:
                    f.write('sensors.{} = {}\n'.format(item_name, item_value))

            f.write('\n')
            attrs = vars(agent)
            for item in sorted(attrs.items()):
                (item_name, item_value) = item
                if item_name not in ["memory", "model", "model_target"]:
                    f.write('agent.{} = {}\n'.format(item_name, item_value))

    def write_log(self, header, value):
        if self.save_log:
            with open(self.log_filename, ('w' if self.n_log == 0 else 'a')) as f:
                writer = csv.writer(f)
                if self.n_log == 0:
                    header.append("time")
                    writer.writerow(header)
                time_str = datetime.datetime.now().strftime("%I:%M:%S %p, %B %d %Y")
                value.append(time_str)
                writer.writerow(value)
                self.n_log += 1