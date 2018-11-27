"""
CNN with 3 conv layers and a fully connected classification layer
"""

import torch
import torch.nn as nn
import logging

from models.MDRNN.MDRNN import MDRNN2D


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Average(nn.Module):
    def forward(self, x):
        x = torch.mean(x, dim=2)
        return x
        

class Sum(nn.Module):
    def forward(self, x):
        x = torch.sum(x, dim=2)
        return x


class Test(nn.Module):
    def forward(self, x):
        x = (x.contiguous).view(x.size(0), x.size(1) * 4, x.size(3), x.size(4))
        return x


class MDRNN_basic(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, output_channels=10, input_channels=3, no_cuda=False, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(MDRNN_basic, self).__init__()

        self.expected_input_size = (32, 32)
        
        # ~ self.bloc1 = nn.Sequential(
            # ~ nn.Conv2d(input_channels, 10, kernel_size=3, stride=1, padding=0),
            # ~ nn.MaxPool2d(kernel_size=2, stride=2),
            # ~ nn.Tanh(),
            # ~ MDRNN2D(10, 20, rnn_type='lstm', no_cuda=no_cuda),
            # ~ Average()
            # ~ Test()
        # ~ )
        
        # ~ self.bloc2 = nn.Sequential(
            # ~ nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=0),
            # ~ nn.MaxPool2d(kernel_size=2, stride=2),
            # ~ nn.Tanh(),
            # ~ MDRNN2D(30, 40, rnn_type='lstm', no_cuda=no_cuda),
            # ~ Average()
            # ~ Test()
        # ~ )
        
        # ~ self.bloc3 = nn.Sequential(
            # ~ nn.Conv2d(40, 50, kernel_size=3, stride=1, padding=0),
            # ~ nn.MaxPool2d(kernel_size=2, stride=2),
            # ~ nn.Tanh(),
            # ~ MDRNN2D(50, 60, rnn_type='lstm', no_cuda=no_cuda),
            # ~ Average()
            # ~ Test()
        # ~ )
        
        # ~ self.fc = nn.Sequential(
            # ~ Flatten(),
            # ~ nn.Linear(240, 128),
            # ~ nn.Linear(128, output_channels)
        # ~ )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 10, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Tanh()
        )
        
        self.mdlstm = MDRNN2D(10, 20, rnn_type='lstm', no_cuda=no_cuda)
        
        self.fc = nn.Linear(80, output_channels)
            

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        
        batch_size = x.size(0)
        
        # ~ x = self.bloc1(x)
        # ~ x = self.bloc2(x)
        # ~ x = self.bloc3(x)
        
        
        
        x = self.conv1(x)
        x, om = self.mdlstm(x)
        
        final_act_1 = om[:, :, 0, -1, -1].view(batch_size, -1)
        final_act_2 = om[:, :, 1, 0, -1].view(batch_size, -1)
        final_act_3 = om[:, :, 2, -1, 0].view(batch_size, -1)
        final_act_4 = om[:, :, 3, 0, 0].view(batch_size, -1)
        x = torch.cat((torch.cat((torch.cat((final_act_1, final_act_2), dim=1), final_act_3), dim=1), final_act_4), dim=1)
        
        x = self.fc(x)
        
        return x
