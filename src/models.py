import data_prep
import eval
import torch 
import util
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

class CNNImageColorizerModel(torch.nn.Module):
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
