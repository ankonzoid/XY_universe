# XY Universe: a 2D particle survival environment

We provide here XY Universe: a 2D particle survival environment for training your Deep Reinforcement Learning agents to stay alive as long as possible by avoiding collisions with "bad" particles. The agent and the "bad" particles are confined to a 2D box, move with fixed constant speed, and experience elastic collisions against the box walls. A Deep Q-Network agent implementation is provided for immediate usage and demonstration of the environment. The agent reward scheme is: +1 for each time step alive, and -100 for "bad" particle collision. The episode termination condition is: agent collision with any "bad" particle. This survival environment was inspired by [Karpathy's Waterworld environment](http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html). 

<p align="center">
<img src="gifs/XY_universe.gif" width="50%">
</p>


### Usage

```
python3 XY_universe.py
```

The output will be placed in the `output` directory, and will consist of agent models (`.h5`) and animations (`.m4`) of the most successful episodes, as well as run logs of the entire simulation (`.csv`, `.pdf`).

### Libraries required

* tensorflow, keras, numpy, scipy, matplotlib, h5py, ffmpeg, tkinter

### Authors

Anson Wong