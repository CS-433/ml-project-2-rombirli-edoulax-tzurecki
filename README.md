# Reinforcement Learning Actor for Blade Pitch Control in Wind Turbine Systems

## Introduction

This project is part of the course CS-433 Machine Learning at EPFL.

The goal of this project is to design a wind turbine pitching control system using reinforcement learning.
It uses a surrogate model to simulate the wind turbine system, and a
Deep Deterministic Policy Gradient (DDPG) agent to optimize the control actions.
A Soft Actor-Critic (SAC) agent is also tested for comparison.

## Directory Structure

```
ML_project2/
    matlab/ # not tracked by git - dataset files should be placed here
        ...
    notebooks/
        outdated-tests/ # those notebooks are not used anymore, not supposed to work, but kept for reference
               ddpg custom implementation.ipynb
               offline ddpg.ipynb
        prepare dataset.ipynb
        surrogate model.ipynb
        test environment.ipynb
        ddpg stablebaselines3.ipynb
    outputs/ # not tracked by git - generated files are saved here
        ... 
    public models/ # pretrained models are placed here, you should copy them to outputs/models
        ...    
    environment.py # the simulation environment, along with useful functions
    mlps.py # functions to generate, load and save MLPs
    parameters.py # parameters for preprocessing, surrogate model and DDPG agent
    README.md # this file
    report.pdf # the report
    requirements.txt # python dependencies 
```

## Setup MATLAB files

To set up the project, you need to download the [MATLAB files](https://drive.switch.ch/index.php/s/zHMJLy50f6b55IF) that contain the wind turbine data.
These files should be placed in the `matlab` as shown in the directory structure below.

```
ML_project2/
    matlab/
        case 1/
            ms008mpt001.mat
            ms008mpt002.mat
            ...
        case 2/
            ms006mpt001.mat
            ms006mpt002.mat
            ...
        case 3/
            vawt_data.mat         
```

## Dataset Preparation

The dataset used in this project is derived from the MATLAB files.
The preprocessing of the data is done in the [prepare dataset](notebooks/prepare%20dataset.ipynb) notebook.
The dataset is only used to train the surrogate model, and is not used in the reinforcement learning algorithm.
The preprocessing steps are as follows:

1. The data is loaded from the MATLAB files and converted into pandas DataFrames with the same column names. The
   dataframes are saved as pickle files in the `outputs/datasets/{case}/dataframes` directory.
2. The data are resampled to 50 steps per revolution using linear interpolation.
3. Angles are converted to have the same units through dataset cases.
4. The state action matrix `X` is created by taking the following rows from the dataframes of the same dataset case.
   - `df[Phase]` &rarr; compute `X[cos_phase]` and `X[sin_phase]`
   - `df[Ct]` &rarr; `X[Ct]`
   - `df[Cr]` &rarr; `X[Cr]`
   - `df[Pitch]` &rarr; `X[pitch]`, compute 
     - `X[dpitch]` : Dpitch is the pitch increment from the previous state to the current state, corresponding approximately to the pitch angular speed.
     - `X[action]` : Action is the pitch increment from the current state to the next state.

5. The labels `Y` are created by taking the next state from the dataframes of the same dataset case. Some features of the next state can be computed at simulation time, so they are not computed by the surrogate model and thus not included in the labels. The only features that are included in the labels are:
   - `df[Ct+1]` &rarr; `Y[Ct]`
   - `df[Cr+1]` &rarr; `Y[Cr]`
6. The data are normalized and shuffled, then saved as a single pickle file in the `outputs/datasets/{case}/array`
   directory.
7. The data from multiple dataset cases are mixed together by randomly selecting experiences through dataset cases using
   a slightly adjusted probability distribution.
8. The data are normalized and shuffled, then saved as a single pickle file in the `outputs/datasets/full/array`
   directory.
   The dataset includes features such as the phase, pitch, and the coefficients of thrust and power.

## Surrogate Model

The surrogate model is a multi-layer perceptron (MLP) neural network that predicts the evolution of the forces `Ct`
and `Cr` based on the current state and the control action.
This is enough to predict the next state of the system, which is used in the simulation environment.
This surrogate model allows us to simulate the system without the need for real access to the wind turbine, and without
having to solve the [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) that describe the system.

The surrogate model is trained in the [surrogate model](notebooks/surrogate%20model.ipynb) notebook. You can also find a
small analysis of the model and the dataset in this notebook, such as a [SHAP](https://shap.readthedocs.io/en/latest/) features importance analysis.

The model's architecture and training parameters can be found in the [parameters](parameters.py) file.
The model is saved in the `outputs/models` directory.

## Environment

The environment is a simulation of the wind turbine system. It uses the surrogate model to predict the next state of the
system based on the current state and the control action. The environment is implemented as a [Gymnasium
](https://gymnasium.farama.org/index.html) environment, making it compatible with reinforcement learning algorithms from libraries like [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/).

The environment is defined in
the [environment](environment.py) file as the `TurbineEnvironment` class extending the `gymnasium.Env` class.

### Test the environment accuracy in simulation

The `simulate_open_loop_episode` function in the [environment](environment.py) file is used to simulate an episode of the system
operation with a given sequence of actions. The function returns the states and rewards for the entire episode.
Giving the same sequence of actions to the environment should produce the same states and rewards.
This is tested in the [test environment](notebooks/test%20environment.ipynb) notebook.

## DDPG Agent

The DDPG agent is a reinforcement learning agent that learns the optimal control policy for the wind turbine system. The
agent is trained in the environment using the DDPG algorithm. The agent's architecture and training parameters can be
found in the [parameters](parameters.py) file. The training process can be monitored using TensorBoard, with the logs
saved in the `outputs/tensorboard` directory.

To train the agent, run the [ddpg stablebaselines3](notebooks/ddpg%20stablebaselines3.ipynb) notebook.
A small agent evaluation is also done in this notebook.

The agent is saved in the `outputs/models` directory.


## SAC Agent

The SAC agent is a reinforcement learning agent that learns the optimal control policy for the wind turbine system. The
agent is trained in the environment using the SAC algorithm. The agent's architecture and training parameters can be
found in the [parameters](parameters.py) file. The training process can be monitored using TensorBoard, with the logs
saved in the `outputs/tensorboard` directory.

To train the agent, run the [sac stablebaselines3](notebooks/sac%20stablebaselines3.ipynb) notebook.
A small agent evaluation is also done in this notebook.

The agent is saved in the `outputs/models` directory.

## Report

The report is available in the [report](report.pdf) file.

## Authors
- [Romain Birling](https://github.com/rombirli)
- [Thomas Zurecki](https://github.com/TZurecki)
- [Edouard Lacroix](https://github.com/edoulaX)