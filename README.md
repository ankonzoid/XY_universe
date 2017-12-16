# XY universe: a 2D particle environment for Deep Reinforcement Learning

We provide here a collection of 2D particle environments for training your agents. Much of the structure is inspired by OpenGym.

## Survival (single-agent environment)

This survival environment consists of a swarm of non-colliding "bad" particles moving abour randomly, and the goal of the agent to stay alive by not touching any of the "bad" particles. This survival environment was inspired by [Karpathy's Waterworld environment](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html) where the goal was to dodge the "bad" particles and eat "good" particles. 

#### Reward scheme: 
* +1 for every time step
* +100 for eating a "good" particle
* -100 for eating a "bad" particle

#### Episode termination condition:
* Upon collision with any "bad" particle

#### Example videos:

<p align="center">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/survival.gif" width="45%">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/waterworld.gif" width="45%">
</p>

## Escape (single-agent environment)

This environment starts with the agent at the left side of the map. The goal of the agent is to make it to the right side of the map as fast as possible without touching any "bad" particles. Unlike the survival environment where the particles move with constant velocity based on their initial conditions, this environment allows for "bad" particles to continually chase the agent. 


#### Reward scheme: 
* -1 for every time step
* dx (x-displacement) reward towards goal

#### Episode termination condition:
* Upon reaching its goal of escaping to the right
* Upon collision with any "bad" particle

#### Example videos:

<p align="center">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/prison_escape.gif" width="45%">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/pass_by.gif" width="45%">
</p>



## Bump (multi-agent environment)

This environment starts with two agents standing at opposite edges of the box, and they are rewarded for x-displacement towards the other edge. There is also small penalty at every time step for not reaching a non-terminal state.

The termination condition for the episode is when any agent reaches its goal, therefore there are two physical mechanisms driving the agent to get to its goal as fast as possible:

1. Existing at each timestep penalizes the agent, therefore early termination is desired

2. The competing agent reaching its goal equates to early termination of the agent in reaching its own goal and collecting as much x-displacement reward


### Usage:

Run the command

> python XY_universe.py

### Libraries required:

* keras, numpy, scipy, matplotlib

### Relevant sources

* Karpathy's Waterworld simulation: http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html

* "Emergent Complexity via Multi-Agent Competition" (Bansal et al): https://arxiv.org/abs/1710.03748

* "Multiagent Cooperation and Competition with Deep Reinforcement Learning" (Tampuu et al): https://arxiv.org/abs/1511.08779

* "Multi-Agent Reinforcement Learning: Independent vs Cooperative Agents" (Tan): https://pdfs.semanticscholar.org/b3fc/56876ad1cdf35ad4af13b991bbb24d219bd9.pdf

### Authors:

Anson Wong