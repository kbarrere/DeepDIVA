import logging

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class _CRNN(nn.Module):
    r"""

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
    conv4 : torch.nn.Sequential
    conv5 : torch.nn.Sequential
        Convolutional layers of the network
    fc1 : torch.nn.Linear
    fc2 : torch.nn.Linear
        Fully connected layer
    cl : torch.nn.Linear
        Final classification fully connected layer
    """

    def __init__(self, output_channels=1000):
        """
        Creates an AlexNet model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        """
        super(_CRNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 4), stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d()
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            Flatten(),
            nn.Linear(325000, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        # Classification layer
        self.cl = nn.Sequential(
            nn.Linear(4096, output_channels),
        )

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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cl(x)
        return x


def crnn(pretrained=False,**kwargs):
    """
    Returns a CRNN model.

    Parameters
    ----------
    """
    model = _CRNN(**kwargs)

    if pretrained:
        logging.info('No pretraining available for this model. Training a network form scratch')

    return model
