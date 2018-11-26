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

        self.mdlstm = MDRNN2D(input_channels, 32, rnn_type='lstm', no_cuda=no_cuda)
        self.fc = nn.Linear(4 * 32, output_channels)

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
        x = self.mdlstm(x)
        final_act_1 = x[:, :, 0, -1, -1].view(batch_size, -1)
        final_act_2 = x[:, :, 1, 0, -1].view(batch_size, -1)
        final_act_3 = x[:, :, 2, -1, 0].view(batch_size, -1)
        final_act_4 = x[:, :, 3, 0, 0].view(batch_size, -1)
        x = torch.cat((torch.cat((torch.cat((final_act_1, final_act_2), dim=1), final_act_3), dim=1), final_act_4), dim=1)
        x = self.fc(x)
        return x
