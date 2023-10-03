import data_prep
import eval
import util
import models

from functools import partial

import os
from os import sep

import torch 
from torch.utils.data import DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler

from dotenv import load_dotenv
load_dotenv()

CNN_MODEL = os.getenv("CNN_MODEL")
MODELS_DIR = os.getenv("MODELS_DIR")


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int) -> float:
    """https://github.com/huggingface/transformers/blob/v4.33.3/src/transformers/optimization.py#L357

    Starts low, and increases factor until reaching num_warmup_steps, where the factor will be 1, 
    then it starts decreasing again.

    Example:

    num_warmup_steps = 100
    num_training_steps = 1000
    
    current_step = 1
    factor = 1 / 100 = 0.01

    current_step = 100
    factor = 1000-100 / 1000-100 = 1.00

    current_step = 400
    factor = 1000-400 / 1000-100 = 600 / 900 = 0.67

    Returns:
        float: Multiplicative factor for learning rate
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def training_loop(ModelType, save_dir, num_epochs=3):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # construct model
    model = ModelType()

    # load data
    train, _test, val = data_prep.create_datasets(device=device)

    # wrap train and val split in dataloader
    # train = Subset(train, range(100))
    train_dataloader = DataLoader(train, batch_size=8, shuffle=False)
    val = Subset(val, range(50))
    eval_dataloader = DataLoader(val, batch_size=8, shuffle=False)

    # "Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters (defined 
    # with torch.nn.Parameter) which are members of the model."
    # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # calculate training_steps and choose 5% warmup steps
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * 0.05)

    # assemble lambda function, which, given current_step during training, 
    # will return a multiplicative factor for the learning rate
    # (copied from HuggingFace source code)
    lr_lambda = partial(_get_linear_schedule_with_warmup_lr_lambda, 
                        num_warmup_steps=num_warmup_steps, 
                        num_training_steps=num_training_steps)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    
    print("Training")
    batch_num = 0
    for epoch in range(num_epochs):
        # loops through batches, so input and target are really
        # lists of inputs and targets
        for input, target in iter(train_dataloader):
            batch_num += 1
            print(f"Batch {batch_num} / {len(train_dataloader)}, Epoch {epoch+1} / {num_epochs}", end="\r")
            
            if batch_num % 100 == 0:
               print()
               metrics = eval.eval_model(model, eval_dataloader, criterion, device)
               print(metrics)

            # move data to GPU
            input, target = input.to(device), target.to(device)
            target_pred = model(input)

            # compute loss
            loss = criterion(target_pred, target)

            # print loss
            if batch_num % 50 == 0:
                print({"training_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # change learning rate
            # (I assume this simply increases "current_step" and calls the lr lambda)
            scheduler.step()

        batch_num = 0

    util.save_model(model, save_dir)

def main():
    training_loop(models.CNNImageColorizerModel, save_dir=f"{MODELS_DIR}{sep}{CNN_MODEL}", num_epochs=3)

if __name__ == "__main__":
    main()