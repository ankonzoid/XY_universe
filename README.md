# XY universe: a 2D particle environment for Deep Reinforcement Learning

We provide here a collection of 2D particle environments for training your agents. Much of the code structure is inspired by OpenGym of partitioning the agent, evironment, and sensors as much as possible.

## Survival (single-agent environment)

This survival environment consists of a bunch of non-colliding "bad" particles moving around randomly, and the goal of the agent is to stay alive by not touching any of the "bad" particles. This survival environment was inspired by [Karpathy's Waterworld environment](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html) where the goal is to dodge the "bad" particles and eat "good" particles. 

#### Reward scheme: 
* +1 for every time step
* +100 for eating a "good" particle
* -100 for eating a "bad" particle

#### Episode termination condition:
* Upon collision with any "bad" particle

#### Example videos:

<p align="center">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/survival1.gif" width="45%">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/survival2.gif" width="45%">
</p>

## Escape (single-agent environment)

This environment starts with the agent at the left side of the box. The goal of the agent is to make it to the right side of the box as fast as possible without touching any "bad" particles. Unlike the survival environment where the particles move with constant velocity based on their initial conditions, this environment allows for "bad" particles to continually chase the agent. 

#### Reward scheme: 
* -1 for every time step
* dx (x-displacement) reward towards goal

#### Episode termination condition:
* Upon reaching its goal of escaping to the right
* Upon collision with any "bad" particle

#### Example videos:

<p align="center">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/escape1.gif" width="45%">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/escape2.gif" width="45%">
</p>


## Bump (multi-agent environment)

This environment starts with two agents standing at opposite edges of the box, and they are rewarded for x-displacement towards the other edge. Although this environment is similar to the Escape environment, there are key differences:

1. There is a multi-agent environment where at each time-step each agent takes an action

2. All particles now perform elastic collisions with eachother.

This environment naturally exhibits competition because the agents should desire early termination towards their respective goals, which in turn will end their competing agent's reward collection prematurely.

#### Reward scheme: 
* -1 for every time step
* dx (x-displacement) reward towards goal

#### Episode termination condition:
* Upon reaching its goal of escaping to the left/right

#### Example videos:

<p align="center">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/bump1.gif" width="45%">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/bump2.gif" width="45%">
</p>


### Usage

Run the command

> python XY_universe.py

### Libraries required

* tensorflow, keras, numpy, scipy, matplotlib, h5py
* ffmpeg, tkinter

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