# XY Universe: a 2D particle environment for training agents

We provide here a 2D XY platform of particle environments for training agents to learn strategies to beat a game.
 
 demonstrate both intuitive and non-trivial tactics that can form under the reinforcement learning framework.

## Environments

### Survival
This survival environment has the simple goal of keeping the agent alive by not letting it get touched by any "bad" particles. This survival environment was inspired by [Karpathy's Waterworld environment](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html) where the goal was to dodge the "bad" particles and eat "good" particles. 

##### Reward system: 
+1 reward every time step

##### Episode termination condition:
Upon collision with any "bad" particle

##### Example video:

<p align="left">
<img src="https://github.com/ankonzoid/XY_universe/blob/master/gifs/survival.gif" width="50%">
</p>


### Bump

This environment starts with two agents standing at opposite edges of the box, and they are rewarded for x-displacement towards the other edge. There is also small penalty at every time step for not reaching a non-terminal state.

The termination condition for the episode is when any agent reaches its goal, therefore there are two physical mechanisms driving the agent to get to its goal as fast as possible:

1) Existing at each timestep penalizes the agent, therefore early termination is desired

2) The competing agent reaching its goal equates to early termination of the agent in reaching its own goal and collecting as much x-displacement reward

Reward system: 
dx reward for x-displacement towards its goal
-1 reward for existence at each timestep

Termination:
Upon reaching its goal


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