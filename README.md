# XY Universe: a 2D particle survival environment

We provide here XY Universe: a 2D particle survival environment for training your Deep Reinforcement Learning agents to stay alive as long as possible by avoiding collisions with "bad" particles. The agent and the "bad" particles are confined to a 2D box, move with random constant velocity, and experience elastic collisions against the 2D box walls. A Deep Q-Network agent implementation is provided for immediate usage and demonstration of the environment. This survival environment was inspired by [Karpathy's Waterworld environment](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html).

<p align="center">
<img src="gifs/XY_universe.gif" width="50%">
</p>

#### Agent reward scheme: 
* +1 for every time step alive
* -100 for colliding into "bad" particle

#### Episode termination condition:
* Agent collision with any "bad" particle

### Usage

```
python3 XY_universe.py
```

The output will be placed in the `output` directory, and will consist of agent models (`.h5`) and animations (`.m4`) of the most successful episodes, as well as run logs of the entire simulation (`.csv`, `.pdf`).

### Libraries required

* tensorflow, keras, numpy, scipy, matplotlib, h5py, ffmpeg, tkinter

### Authors

Anson Wong