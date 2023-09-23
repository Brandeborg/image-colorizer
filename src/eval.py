
import torch
from torch import Tensor, uint8
from torch.nn import Module

from torch.utils.data import DataLoader

def accuracy(pred: Tensor, target: Tensor) -> float:
    """Calculate accuracy between prediction image and target image, 
    defined by how many "pixels" in the same position were exactly the same.

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

def eval_model(model: Module, eval_dataloader: DataLoader, criterion, device: str) -> dict:
    """Evaluate model performence on dataset set

    Args:
        model (Module): Model to evaluate
        eval_dataloader (DataLoader): A dataloader containing test og validation dataset
        criterion (_type_): The loss function
        device (str): Device (cuda, cpu, etc.)

    Returns:
        dict: A dict containing loss and accuracy
    """
    model.eval()

    metrics = {"eval_loss": 0, "eval_accuracy": 0}

    batch_num = 0

    print("Evaluating dataset")
    with torch.no_grad():
        for input, target in iter(eval_dataloader):
            batch_num += 1
            print(f"Batch {batch_num} / {len(eval_dataloader)}", end="\r")

            # move data to GPU
            input, target = input.to(device), target.to(device)
            target_pred = model(input)

            metrics["eval_loss"] += criterion(target_pred, target)
            
            for sample_input, sample_target in zip(input, target):
                sample_target_pred = model(torch.unsqueeze(sample_input, 0))

                metrics["eval_accuracy"] += accuracy(sample_target_pred, sample_target)

    metrics["eval_loss"] = metrics["eval_loss"].item() / len(eval_dataloader)
    metrics["eval_accuracy"] = metrics["eval_accuracy"] / len(eval_dataloader) * eval_dataloader.batch_size

    model.train()
    return metrics
