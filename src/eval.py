
import data_prep
import util

import os
from os import sep

import torch
from torch import Tensor, uint8
from torch.nn import Module
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

CNN_MODEL = os.getenv("CNN_MODEL")
MODELS_DIR = os.getenv("MODELS_DIR")

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
    # set model to eval mode
    model.eval()

    metrics = {"eval_loss": 0, "eval_accuracy": 0}

    batch_num = 0

    print("Evaluating dataset")
    # using no_grad scope becuase gradients don't need to be calculated for eval, saves time
    with torch.no_grad():
        for input, target in iter(eval_dataloader):
            batch_num += 1
            print(f"Batch {batch_num} / {len(eval_dataloader)}", end="\r")

            # move data to GPU
            input, target = input.to(device), target.to(device)
            target_pred = model(input)

            metrics["eval_loss"] += criterion(target_pred, target)
            
            # accuracy function doesn't do batches right now, 
            # so loop through batch
            for sample_input, sample_target in zip(input, target):
                sample_target_pred = model(torch.unsqueeze(sample_input, 0))

                metrics["eval_accuracy"] += accuracy(sample_target_pred, sample_target)

    metrics["eval_loss"] = metrics["eval_loss"].item() / len(eval_dataloader)
    metrics["eval_accuracy"] = metrics["eval_accuracy"] / (len(eval_dataloader) * eval_dataloader.batch_size)

    # back to training mode, in case the eval step was done between epochs
    model.train()
    return metrics

def display_result(input, output, target):
    # result
    input = input[0].permute(1,2,0).cpu()
    plt.subplot(1,3,1)
    plt.imshow(input, cmap="gray")

    output = output[0].permute(1,2,0).cpu()
    plt.subplot(1,3,2)
    plt.imshow(output)
    
    target = target.permute(1,2,0)
    plt.subplot(1,3,3)
    plt.imshow(target)

    plt.show()

def test_img():
    # passes example through trained model and illustrates result
    # load data
    _train, test, _val = data_prep.create_datasets(device=device)
    model = util.load_model(f"{MODELS_DIR}{sep}{CNN_MODEL}")

    # wrap test split in dataloader
    test_dataloader = DataLoader(test, batch_size=2, shuffle=False)

    inputs, targets = next(iter(test_dataloader))

    # model expects batches on GPU, so move to GPU and unsqueeze("wrap in batch")
    input = inputs[1].to(device).unsqueeze(0)

    output = (model(input) * 255).detach().to(torch.uint8)

    # result
    display_result(input, output, targets[1])

def main():
    test_img()

if __name__ == "__main__":
    main()