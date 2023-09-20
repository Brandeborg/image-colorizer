
from torch import Tensor, uint8

def accuracy(pred: Tensor, target: Tensor) -> float:
    """Calculate accuracy between prediction image and target image, defined by how many "pixels" in the same position were exactly the same.

    Args:
        pred (Tensor): Predicted tensor image
        target (Tensor): Target tensor image

    Returns:
        float: The accuracy as a number between 0. and 1.
    """

    # transform result, consisting of floats between 0 and 1, to flat tensor consisting of ints between 0 and 255
    pred = (pred * 255).detach().to(uint8).flatten()
    target = (target * 255).detach().to(uint8).flatten()

    # make list of true/false predictions
    correct = pred == target

    # calcuate percentage of true predictions
    accuracy = correct.sum().item() / correct.size(dim=-1)

    return accuracy