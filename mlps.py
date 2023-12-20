import json
import os
import zipfile

import torch.nn as nn

from parameters import *


def create_mlp(widths: tuple, seed: int = 0) -> nn.Sequential:
    """
    Create multi-layer perceptron for the given layers widths
    :param widths: tuple[int], the layer widths
    :param seed: int, seed to use
    :return: the generated MLP
    """
    # set seed for reproducibility
    torch.manual_seed(seed)
    layers = [
        layer
        for w1, w2 in zip(widths, widths[1:-1])
        for layer in [nn.Linear(w1, w2), nn.BatchNorm1d(w2), nn.ReLU()]
    ]
    layers.append(nn.Linear(widths[-2], widths[-1]))
    return nn.Sequential(*layers)


def save_model(model, model_name, infos=None):
    """
    Save model to a zip file on disk
    :param model: pytorch module to save
    :param model_name: name of the model (stemmed path)
    :param infos: dict, additional infos to save
    :return: None
    """
    if infos is None:
        infos = {}
    # Save state dictionary
    torch.save(model.state_dict(), f'{model_name}.pt')

    # Save infos
    with open(f'{model_name}_infos.txt', 'w') as f:
        json.dump(infos, f, indent=4)

    # Write everything to a zip file
    with zipfile.ZipFile(f'{model_dir}/{model_name}.zip', 'w') as zipf:
        zipf.write(f'{model_name}.pt')
        zipf.write(f'{model_name}_infos.txt')

    # Remove temporary files
    os.remove(f'{model_name}.pt')
    os.remove(f'{model_name}_infos.txt')


def load_model(model_name, device):
    """
    Load model from disk
    :param model_name: name of the model (stemmed path)
    :return: pytorch module
    """
    # Unzip model
    with zipfile.ZipFile(model_dir / model_name, 'r') as zipf:
        zipf.extractall()
    model_name = model_name[:-4]  # remove zip suffix

    # Load state dictionary
    state_dict = torch.load(f'{model_name}.pt', map_location=device)

    # Load model
    model = create_mlp(surrogate_model_layer_widths).to(device)

    # Remove temporary files
    os.remove(f'{model_name}.pt')
    os.remove(f'{model_name}.pkl')
    if os.path.exists(f'{model_name}_infos.txt'):
        os.remove(f'{model_name}_infos.txt')

    model.load_state_dict(state_dict)
    return model
