import logging

import torch.nn as nn
import torch

from models.MDRNN.MDRNN import MDRNN2D


class GetOutputMap(nn.Module):
    def forward(self, x):
        x = x[1]
        return x


class Average(nn.Module):
    def forward(self, x):
        x = torch.mean(x, dim=2)
        return x


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


class MDLSTM_RNN(nn.Module):
    
    def __init__(self, output_channels=80, expected_input_size=(128, 2174), no_cuda=False):
        
        super(MDLSTM_RNN, self).__init__()
        
        self.expected_input_size = expected_input_size
        
        n=8
        max_hidden_unit=120
        input_channels=3
        
        self.bloc1 = nn.Sequential(
            nn.Conv2d(input_channels, min(max_hidden_unit, 1*n), kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            MDRNN2D(min(max_hidden_unit, 1*n), min(max_hidden_unit, 2*n), dropout=0.25, rnn_type='lstm', no_cuda=no_cuda),
            GetOutputMap(),
            Average()
        )
        
        self.bloc2 = nn.Sequential(
            nn.Conv2d(min(max_hidden_unit, 2*n), min(max_hidden_unit, 3*n), kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Dropout2d(p=0.25),
            MDRNN2D(min(max_hidden_unit, 3*n), min(max_hidden_unit, 4*n), dropout=0.25, rnn_type='lstm', no_cuda=no_cuda),
            GetOutputMap(),
            Average()
        )
        
        self.bloc3 = nn.Sequential(
            nn.Conv2d(min(max_hidden_unit, 4*n), min(max_hidden_unit, 5*n), kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.Dropout2d(p=0.25),
            MDRNN2D(min(max_hidden_unit, 5*n), min(max_hidden_unit, 6*n), dropout=0.25, rnn_type='lstm', no_cuda=no_cuda),
            GetOutputMap(),
            Average()
        )
        
        self.bloc4 = nn.Sequential(
            nn.Conv2d(min(max_hidden_unit, 6*n), min(max_hidden_unit, 7*n), kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout2d(p=0.25),
            MDRNN2D(min(max_hidden_unit, 7*n), min(max_hidden_unit, 8*n), dropout=0.25, rnn_type='lstm', no_cuda=no_cuda),
            GetOutputMap(),
            Average()
        )
        
        self.bloc5 = nn.Sequential(
            nn.Conv2d(min(max_hidden_unit, 8*n), min(max_hidden_unit, 9*n), kernel_size=(4, 2), stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout2d(p=0.25),
            MDRNN2D(min(max_hidden_unit, 9*n), min(max_hidden_unit, 10*n), dropout=0.25, rnn_type='lstm', no_cuda=no_cuda),
            GetOutputMap(),
            Average()
        )
        
        self.features_size = (((self.expected_input_size[0] - 2) // 2 - 2) // 2 - 2 ) // 2 - 5, (((self.expected_input_size[0] - 2) // 2 - 2) // 2 - 2) // 2 - 3
        
        self.collapse = Collapse(height=self.features_size[0], width=1)
        
        self.fc = nn.Linear(min(max_hidden_unit, 10*n), output_channels)
            
        
    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = self.bloc4(x)
        x = self.bloc5(x)
        
        x = self.collapse(x)
        
        x = x.permute(0, 2, 1)
        
        x = self.fc(x)
        
        return x


def mdlstm_rnn(pretrained=False,**kwargs):
    """
    Returns a CRNN model.

    Parameters
    ----------
    """
    model = MDLSTM_RNN(**kwargs)

    if pretrained:
        logging.info('No pretraining available for this model. Training a network form scratch')

    return model
