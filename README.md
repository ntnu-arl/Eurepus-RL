# Eurepus-RL
Reinforcement Learning Policy Training for the In-Flight Attitude Control of the Eurepus Quadruped.

# Installation and Running
This project is based on the [omniisaacgymenvs](https://github.com/isaac-sim/OmniIsaacGymEnvs) repository. All necessary steps to run the training are explained in detail there. The essential steps are reiterated here briefly.

1. Install the omniisaacgymenvs dependencies as described in the "Installation" section of the README in the omniisaacgymenvs repository.
2. Run the training with `PYTHON_PATH train.py headless=True/False enable_livestream=True/False test=True/False`

# Important Config Parameters
In the task config:
- `env/control/`: Change important motor model parameters, like stiffness and damping.
- `env/learn/rewards/`: Change reward weighting.
- `env/learn/episodeLength_s/`: Change episode length.

In the train config:
- `params/network/mlp`: Change network parameters, like size and activation.
- `params/config`: Change RL-Games PPO specific parameters, like horizon length.

# USD
The USD of Eurepus can be found in the USD folder.

# Hardware Files and Control Code
You can find the following files at [Eurepus-design](https://github.com/ntnu-arl/Eurepus-design): 
- CAD files of Eurepus (SOLIDWORKS).
- The deployed controller code for Eurepus.
- CAD files for the Free Fall and Rotating Pole test setups.

# WIP
This repository is a work in progress. Efforts will be made to restructure it before publishing it sometime during the fall.
