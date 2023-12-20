from pathlib import Path

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42

# files
project_directory = Path(__file__).parent
matlab_directory = project_directory / 'matlab'
output_directory = project_directory / 'outputs'
dataset_directory = output_directory / 'dataset'
dataset_directory.mkdir(parents=True, exist_ok=True)
model_dir = output_directory / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

cases = [x.stem for x in matlab_directory.iterdir() if x.is_dir()]
matlab_cases_dir = [matlab_directory / case for case in cases]
dataset_cases_dir = [dataset_directory / case for case in cases]

# dataset
dataset_types = 'phavg', 'res'
x_columns = 'cos_phase', 'sin_phase', 'Ct', 'Cr', 'pitch', 'dpitch', 'action'
y_columns = 'Ct', 'Cr'

# surrogate model
surrogate_model_layer_widths = len(x_columns), 16, 32, 64, 128, 256, 512, 32, len(y_columns)
surrogate_model_learning_rate = 1e-3
surrogate_model_regularizer = 1e-3
surrogate_model_scheduler_step_size = 10
surrogate_model_scheduler_gamma = 0.95
surrogate_model_epochs = 1000
surrogate_model_batch_size = 2048
surrogate_model_test_split = 0.1

# best surrogate model checkpoint
best_surrogate_model_checkpoint = 'surrogate_model_res-full-2023.12.15_23.01.38.zip'

# environment
turn_length = 50  # resample datasets to this value using linear interpolation
training_n_turns = 3  # use this many turns per episode to learn good long term behavior
agent_training_n_turns = 1  # number of turns to train the agent
rad_per_step = 0.12615061515438047  # radians per step
# clip angle and pitch to reproduce the physical limits of the system
pitch_bounds = np.deg2rad(-45), np.deg2rad(45)

# DDPG agent
ddpg_total_timesteps = 20000
ddpg_learning_rate = 1e-4

# SAC agent
sac_total_timesteps = 10000
sac_learning_rate = 1e-2
