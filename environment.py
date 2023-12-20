import json

import gymnasium
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from mlps import load_model
from parameters import output_directory, y_columns, x_columns, device, best_surrogate_model_checkpoint, rad_per_step, \
    pitch_bounds, turn_length, training_n_turns


class StateEncoder:
    """
    This class contains infos about the environment. (norms, bounds for state and action)
    """

    def __init__(self, case='full', dataset_type='phavg'):
        # load normalization infos and dataset
        self.norm_infos = json.load((output_directory / 'dataset' / case / 'array' / dataset_type / 'norm.json').open())
        self.x_means_tensor = torch.tensor(self.norm_infos['X mean'], dtype=torch.float32).to(device)
        self.x_stds_tensor = torch.tensor(self.norm_infos['X std'], dtype=torch.float32).to(device)
        self.y_means_tensor = torch.tensor(self.norm_infos['Y mean'], dtype=torch.float32).to(device)
        self.y_stds_tensor = torch.tensor(self.norm_infos['Y std'], dtype=torch.float32).to(device)
        self.X = np.load(output_directory / 'dataset' / case / 'array' / dataset_type / 'X.npy')
        self.X = torch.tensor(self.X, dtype=torch.float32).to(device)

        # denormalize
        self.X = self.X * self.x_stds_tensor + self.x_means_tensor
        self.min_x, self.max_x = self.X.min(dim=0)[0], self.X.max(dim=0)[0]

        # infos for environment
        self.min_state = self.min_x[:-1]
        self.max_state = self.max_x[:-1]
        self.min_action = self.min_x[-1].item()
        self.max_action = self.max_x[-1].item()
        self.initial_states = self.X[0, :-1].reshape(1, -1)


class TurbineEnvironment(gymnasium.Env):
    """
    This class is the environment for the agent. It is a gymnasium environment.
    """

    def __init__(self, case='full', dataset_type='res',
                 surrogate_model: str = best_surrogate_model_checkpoint):
        """
        Initialize the environment.
        :param case: dataset case type (case 1, case 2, case 3, full)
        :param dataset_type: dataset type (phavg, res)
        :param surrogate_model: surrogate model checkpoint name
        """
        self.surrogate_model = load_model(surrogate_model, device)
        self.surrogate_model.eval()
        self.state_encoder = StateEncoder(case, dataset_type)
        self.action_space = gymnasium.spaces.Box(low=self.state_encoder.min_action, high=self.state_encoder.max_action,
                                                 shape=(1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=self.state_encoder.min_state.cpu().numpy(),
                                                      high=self.state_encoder.max_state.cpu().numpy(),
                                                      shape=(6,), dtype=np.float32)
        self.state = None
        self.step_counter = 0

    def reset(self, *args, **kwargs):
        """
        Reset the environment for a new episode.
        :param args: Ignored
        :param kwargs: Ignored
        :return: initial state, empty info dict
        """
        self.step_counter = 0
        self.state = self.state_encoder.initial_states[np.random.randint(0, len(self.state_encoder.initial_states))]
        return self.state.cpu().numpy(), {}

    def step(self, action):
        """
        Perform a step in the environment.
        :param action: action to perform
        :return: observation, reward, terminated, truncated, info
        """
        with torch.no_grad():
            state = self.state.clone().detach()

            # merge state (tensor) and action (float)
            action = torch.tensor(action).to(device).view(-1)
            x = torch.cat((state, action))
            # normalize x
            x = (x - self.state_encoder.x_means_tensor) / self.state_encoder.x_stds_tensor
            # predict Ct and Cr
            y = self.surrogate_model(x.view(1, -1).float())
            # denormalize y
            y = y * self.state_encoder.y_stds_tensor + self.state_encoder.y_means_tensor
            ct, cr = y[:, y_columns.index('Ct')], y[:, y_columns.index('Cr')]

            # replace in state
            state[x_columns.index('Ct')] = ct
            state[x_columns.index('Cr')] = cr

            # compute new phase
            phase = torch.arctan2(state[x_columns.index('sin_phase')], state[x_columns.index('cos_phase')]) + \
                    rad_per_step

            # replace in state
            state[x_columns.index('sin_phase')] = torch.sin(phase)
            state[x_columns.index('cos_phase')] = torch.cos(phase)

            # compute new pitch
            pitch = state[x_columns.index('pitch')] + action
            truncated = pitch_bounds[0] > pitch or pitch > pitch_bounds[1]

            pitch = torch.clamp(pitch, *pitch_bounds)
            # replace in stae
            state[x_columns.index('pitch')] = pitch

            # put action in dpitch place
            state[x_columns.index('dpitch')] = action

            # compute cp = radius * omega * ct and thrust forces (ignore radius, no impact)
            reward = rad_per_step * ct

            # free memory
            del x, y, ct, cr, phase, pitch
            torch.cuda.empty_cache()  # Explicitly release GPU memory
            self.step_counter += 1
            self.state = state

            observation = state.cpu().numpy()  # todo : remove phase from here ?
            terminated = (self.step_counter >= turn_length * training_n_turns)
            info = {}

            return observation, reward.cpu(), terminated, truncated.cpu(), info


def simulate_open_loop_episode(env: TurbineEnvironment, actions, initial_state):
    """
    Simulate an episode with the given actions and initial state.
    :param env: The environment to use
    :param actions: Actions to perform
    :param initial_state: Initial state
    :return: states, rewards
    """
    states = []
    rewards = []
    env.reset()
    env.state = initial_state
    states.append(initial_state.cpu())
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        states.append(obs)
        rewards.append(reward)
    return np.array(states).T, np.array(rewards)


def simulate_closed_loop_episode(env: TurbineEnvironment, model, initial_state, n=200):
    """
    Simulate an episode with the given model and initial state.
    :param env: The environment to use
    :param model: The model to use
    :param initial_state: Initial state
    :param n: Number of steps to simulate
    :return: states, actions, rewards
    """
    states = []
    rewards = []
    actions = []
    env.reset()
    env.state = initial_state
    states.append(initial_state.cpu())
    for _ in range(n):
        action, _states = model.predict(states[-1])
        obs, reward, done, truncated, info = env.step(action)
        states.append(obs)
        rewards.append(reward)
        actions.append(action)
    return np.array(states).T, np.array(actions), np.array(rewards)


def load_test_episode(case: str = 'case 1', dataset_type: str = 'res', i: int = 0):
    """
    Load a test episode from the dataset.
    :param case: The dataset case to use (case 1, case 2, case 3)
    :param dataset_type: The dataset type to use (phavg, res)
    :param i: The experience index to use within the dataset case
    :return: states, initial_state, actions
    """
    dir = output_directory / 'dataset' / case / 'dataframes' / dataset_type
    files = [x for x in dir.iterdir() if x.is_file()]
    file = files[i]
    test_df = pd.read_pickle(file)
    actions = np.array(test_df['pitch'])[1:] - np.array(test_df['pitch'])[:-1]
    phase = test_df['phase'].to_numpy().T
    cos_phase = np.cos(np.deg2rad(phase))
    sin_phase = np.sin(np.deg2rad(phase))
    dpitch = actions
    states = pd.DataFrame(columns=x_columns[:-1])
    states['dpitch'] = dpitch
    states['pitch'] = test_df['pitch'][:-1]
    states['cos_phase'] = cos_phase[:-1]
    states['sin_phase'] = sin_phase[:-1]
    states['Ct'] = test_df['Ct'][:-1]
    states['Cr'] = test_df['Cr'][:-1]
    states = states.to_numpy().T
    initial_state = torch.tensor(states[:, 0]).float().to(device)
    return states, initial_state, actions


def plot_compare_episodes(pred_states,
                          pred_actions,
                          true_states,
                          true_actions,
                          n: int = int(1e9)):
    """
    Plot the predicted and true states and actions. (4 plots : pred, true, pred and true, pred - true)
    :param pred_states: predicted states by a simulator
    :param pred_actions: predicted actions by a model
    :param true_states: true states recorded in the dataset
    :param true_actions: true actions recorded in the dataset
    :param n: number of points to plot
    :return: None
    """

    def plot_feature(pred_feature, true_feature):
        m = min(n, len(pred_feature), len(true_feature))
        pred_feature = pred_feature[:m].reshape(-1)
        true_feature = true_feature[:m].reshape(-1)
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].set_title('pred')
        axs[0].plot(pred_feature)
        axs[1].set_title('true')
        axs[1].plot(true_feature)
        axs[2].set_title('pred and true')
        axs[2].plot(pred_feature, label='pred')
        axs[2].plot(true_feature, label='true')
        axs[2].legend()
        axs[3].set_title('pred - true')
        axs[3].plot(np.array(pred_feature) - np.array(true_feature))
        return fig

    # plot actions
    fig = plot_feature(pred_actions, true_actions)
    fig.suptitle('action')
    plt.show()
    # plot features
    for feature in 'Ct', 'Cr', 'pitch', 'cos_phase', 'sin_phase':
        i = x_columns.index(feature)
        fig = plot_feature(pred_states[i], true_states[i])
        fig.suptitle(feature)
        plt.show()


if __name__ == '__main__':
    # little crash test
    env = TurbineEnvironment()
    env.reset()
    array = []
    for i in range(50):
        array.append(env.step(0))
    print('ok')
    print(array)
