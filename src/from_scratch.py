import data_prep
import eval
import torch 
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

class ImageColorizerModel(torch.nn.Module):
    """
    Based on:
    https://medium.com/mlearning-ai/building-an-image-colorization-neural-network-part-4-implementation-7e8bb74616c
    """
    def __init__(self):
        """Follows a U-net architecture, where previous layer outputs are passed to future layers,
        to avoid losing important data.
        """
        super().__init__()
        # down conv
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)

        # Dilation layers (conv layer that "skips pixels")
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv6_bn = nn.BatchNorm2d(256)

        # up conv
        self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) 
        self.t_conv1_bn = nn.BatchNorm2d(128)
        # the in-channel is twice the previous out-channel 
        # becuase the result is concatted with output of self.conv3(x_2) (see forward)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(64)
        self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv3_bn = nn.BatchNorm2d(32)
        self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.output = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Makes the forward pass through the network
        """
        x_1 = F.relu(self.conv1_bn(self.conv1(x)))
        x_2 = F.relu(self.conv2_bn(self.conv2(x_1)))
        x_3 = F.relu(self.conv3_bn(self.conv3(x_2)))
        x_4 = F.relu(self.conv4_bn(self.conv4(x_3)))

        # Dilation layers.
        x_5 = F.relu(self.conv5_bn(self.conv5(x_4)))
        x_5_d = F.relu(self.conv6_bn(self.conv6(x_5)))


        x_6 = F.relu(self.t_conv1_bn(self.t_conv1(x_5_d)))

        # concatenate the x_3 to x_6, doubling the amount of channels
        x_6 = torch.cat((x_6, x_3), 1)

        x_7 = F.relu(self.t_conv2_bn(self.t_conv2(x_6)))
        x_7 = torch.cat((x_7, x_2), 1)
        x_8 = F.relu(self.t_conv3_bn(self.t_conv3(x_7)))
        x_8 = torch.cat((x_8, x_1), 1)
        x_9 = F.relu(self.t_conv4(x_8))
        x_9 = torch.cat((x_9, x), 1)
        x = self.output(x_9)
        return x

def training_loop(num_epochs=3):
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # construct model
    model = ImageColorizerModel()

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

training_loop()

