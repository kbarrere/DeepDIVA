import logging

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class Collapse(nn.Module):
    """
    Collecting features among the horizontal axis
    Max pooling on each column
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

    collapse : torch.nn.MaxPool2d
        Collecting features among the horizontal axis
        Max pooling on each column

    lstm : torch.nn.LSTM
        Stacked BLSTMs layers

    fc : torch.nn.Linear
        Fully connected layers that take the outputs of each LSTM cell
        and convert it to characters

    """

    def __init__(self, output_channels=80, expected_input_size=(128, 5248), num_lstm=3):
        """
        Creates a CRNN model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of characters possible to produce
            Default is 61
        expected_input_size : int x int
            The size in pixels of the input images
            It is used to compute the features size and intiate the collaspe layer
        """
        super(_CRNN, self).__init__()

        self.expected_input_size = expected_input_size
        self.num_lstm = num_lstm

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
            nn.MaxPool2d(kernel_size=2, stride=2),
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

        self.features_size = (((self.expected_input_size[0] - 2) // 2 - 2) // 2 - 2 ) // 2 - 5, (((self.expected_input_size[0] - 2) // 2 - 2) // 2 - 2) // 2 - 3
        
        self.collapse = Collapse(height=self.features_size[0], width=1)
        
        self.lstm = nn.LSTM(128, 128, num_layers=self.num_lstm, batch_first=True, dropout=0.5, bidirectional=True)

        self.fc = nn.Linear(256, output_channels)


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

        # Max pooling in each column of the feature image
        x = self.collapse(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = x.contiguous()

        x = self.fc(x)

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

def crnn2(pretrained=False,**kwargs):
    """
    Returns a CRNN model.

    Parameters
    ----------
    """
    model = _CRNN(num_lstm=2, **kwargs)

    if pretrained:
        logging.info('No pretraining available for this model. Training a network form scratch')

    return model

def crnn3(pretrained=False,**kwargs):
    """
    Returns a CRNN model.

    Parameters
    ----------
    """
    model = _CRNN(num_lstm=3, **kwargs)

    if pretrained:
        logging.info('No pretraining available for this model. Training a network form scratch')

    return model

def crnn4(pretrained=False,**kwargs):
    """
    Returns a CRNN model.

    Parameters
    ----------
    """
    model = _CRNN(num_lstm=4, **kwargs)

    if pretrained:
        logging.info('No pretraining available for this model. Training a network form scratch')

    return model

def crnn5(pretrained=False,**kwargs):
    """
    Returns a CRNN model.

    Parameters
    ----------
    """
    model = _CRNN(num_lstm=5, **kwargs)

    if pretrained:
        logging.info('No pretraining available for this model. Training a network form scratch')

    return model

