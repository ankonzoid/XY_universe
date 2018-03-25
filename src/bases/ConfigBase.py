"""

 ConfigBase.py  (author: Anson Wong / git: ankonzoid)


 Copyright 2018 Anson Wong

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
from abc import ABC, abstractmethod

class ConfigBase(ABC):

    @abstractmethod
    def model_filename(self):
        pass

    @abstractmethod
    def animation_filename(self):
        pass

    @abstractmethod
    def write_settings(self):
        pass

    @abstractmethod
    def write_log(self):
        pass