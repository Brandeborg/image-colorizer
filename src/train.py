import data_prep
import eval
import torch 
import util
import models

import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


def training_loop(ModelType, save_dir, num_epochs=3):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # construct model
    model = ModelType()

    # load data
    train, _test, val = data_prep.create_datasets(device=device)

    # wrap train and test split in dataloader
    train_dataloader = DataLoader(train, batch_size=8, shuffle=False)
    eval_dataloader = DataLoader(val, batch_size=8, shuffle=False)

    # "Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters (defined 
    # with torch.nn.Parameter) which are members of the model."
    # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # loops through batches, so input and target are really
    # lists of inputs and targets

    print("Training")
    batch_num = 0
    for epoch in range(num_epochs):
        for input, target in iter(train_dataloader):
            batch_num += 1
            print(f"Batch {batch_num} / {len(train_dataloader)}, Epoch {epoch+1} / {num_epochs}", end="\r")
            
            if batch_num % 10 == 0:
               print()
               metrics = eval.eval_model(model, eval_dataloader, criterion, device)
               print(metrics)

            # move data to GPU
            input, target = input.to(device), target.to(device)
            target_pred = model(input)

            # compute loss
            loss = criterion(target_pred, target)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    util.save_model(model, save_dir)

    # TODO: move somewhere else
    # passes example through trained model and illustrates result
    inputs, targets = next(iter(train_dataloader))
    input = inputs[0].to(device).unsqueeze(0)

    output = (model(input) * 255).detach().to(torch.uint8)

    # result
    display_result(input, output, targets[0])

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


def main():
    training_loop(models.CNNImageColorizerModel, save_dir="cnn_image_colorizer_from_scratch")

if __name__ == "__main__":
    main()