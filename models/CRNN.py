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


class Collapse(nn.Module):
    """
    TODO
    """
    def __init__(self, height=23, width=1):
        super(Collapse, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=(height, width))

    def forward(self, x):
        x = self.mp(x)
        x = x.view(x.size()[0], x.size()[1], -1)
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

    collapse :
        Collecting features among the horizontal axis



    """

    def __init__(self, output_channels=61):
        """
        Creates an AlexNet model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        """
        super(_CRNN, self).__init__()

        self.expected_input_size = (64, 64)

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(8)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(32)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(64)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 2), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(128)
        )

        #self.collapse = Collapse(height=11, width=1) #words
        self.collapse = Collapse(height=23, width=1) #lines
        
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)

        self.fc1 = nn.Sequential(
            nn.Linear(256, output_channels)
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

        x = self.collapse(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = x.contiguous()

        x = self.fc1(x)

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
