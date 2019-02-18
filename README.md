# XY universe: a 2D particle environment for Deep Reinforcement Learning

We provide here a collection of 2D particle environments for training your agents. Much of the code structure is inspired by OpenGym of partitioning the agent, evironment, and sensors as much as possible.

## Survival (single-agent environment)

This survival environment consists of a bunch of non-colliding "bad" particles moving around randomly, and the goal of the agent is to stay alive by not touching any of the "bad" particles. This survival environment was inspired by [Karpathy's Waterworld environment](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html) where the goal is to dodge the "bad" particles and eat "good" particles. Our two settings is "survive_1" which contains 110 bad particles, and "survive_2" which contains 55 good particles nad 55 bad particles.

#### Reward scheme: 
* +1 for every time step
* +100 for eating a "good" particle
* -100 for eating a "bad" particle

#### Episode termination condition:
* Upon collision with any "bad" particle

#### Example videos:

<p align="center">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/survival_bad.gif" width="45%">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/survival_bad_good.gif" width="45%">
</p>

### Usage

Run the command

> python XY_universe.py

### Libraries required

* tensorflow, keras, numpy, scipy, matplotlib, h5py
* ffmpeg, tkinter

### Notes:

MacOS might need you to create: ~/.matplotlib/matplotlibrc and add the line "backend: tkAgg"

### Relevant resources

* [Karpathy's Waterworld simulation](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html)

* ["Emergent Complexity via Multi-Agent Competition" (Bansal et al)](https://arxiv.org/abs/1710.03748)

* ["Multiagent Cooperation and Competition with Deep Reinforcement Learning" (Tampuu et al)](https://arxiv.org/abs/1511.08779)

* ["Multi-Agent Reinforcement Learning: Independent vs Cooperative Agents" (Tan)](https://pdfs.semanticscholar.org/b3fc/56876ad1cdf35ad4af13b991bbb24d219bd9.pdf)

* ["Demystifying Deep Reinforcement Learning"](https://ai.intel.com/demystifying-deep-reinforcement-learning/)

* ["Let's Make an A3C: Theory"](https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/)

* ["Let's Make an A3C: Implementation"](https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/)

### Authors:

Anson Wong