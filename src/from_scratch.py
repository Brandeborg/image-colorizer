import data_prep
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ImageColorizerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 3, (5,2), padding="same"),
            torch.nn.Conv2d(3, 3, (3,2), padding="same")])

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = F.relu(x)

        return x

def training_loop():
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # construct model
    model = ImageColorizerModel()

    # load data
    dataloader = data_prep.create_dataloader()

    # "Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters (defined 
    # with torch.nn.Parameter) which are members of the model."
    # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-optim
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    # loops through batches, so input and target are really
    # lists of inputs and targets

    iters = 0
    for input, target in iter(dataloader):
        if iters == 500:
            break

        # move data to GPU
        input, target = input.to(device), target.to(device)
        target_pred = model(input)

        # compute loss
        loss = criterion(target_pred, target)
        print(loss)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iters += 1

    inputs, targets = next(iter(dataloader))
    input = inputs[0].to(device)

    print (input)
    output = (model(input) * 255).detach().to(torch.uint8).cpu()
    print(output)
    output = output.permute(1,2,0)

    plt.imshow(output)
    plt.show()

training_loop()

