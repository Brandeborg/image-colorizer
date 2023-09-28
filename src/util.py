from models import CNNImageColorizerModel
import torch

from os import sep
import os

def save_model(model: CNNImageColorizerModel, dir_path: str):
    """Saves state_dict of ImageColorizerModel at dir_path

    Args:
        model (ImageColorizerModel): The model to save
        dir_path (str): Where to save the model
    """

    # make dirs if they do not exist
    parent_dirs = f"{sep}".join(dir_path.split(sep)[:-1])

    if not os.path.exists(parent_dirs):
        os.makedirs(parent_dirs)

    # save
    torch.save(model.state_dict(), dir_path)

def load_model(dir_path: str):
    """Loads an ImageColorizerModel

    Args:
        dir_path (str): Where the model is located
    """
    model = CNNImageColorizerModel()
    model.load_state_dict(torch.load(dir_path))
    model.eval()

    return model