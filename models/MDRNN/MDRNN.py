"""
Code from
https://github.com/exe1023/MDRNN
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.MDRNN.rnn import MDGRU, MDLSTM

use_cuda = torch.cuda.is_available()

class RNN(nn.Module):
    '''
    Wrapped RNN
    '''
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 layer_norm=False, 
                 dropout=0,
                 rnn_type='gru'):
        super(RNN, self).__init__()
        # these two paramters are now fixed (not implemented),
        # but we may need them?
        self.bidirectional = False
        self.num_layers = 1

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = MDGRU(input_size, 
                             hidden_size, 
                             layer_norm=layer_norm)
        elif rnn_type =='lstm':
            self.rnn = MDLSTM(input_size, 
                              hidden_size, 
                              dropout=dropout,
                              layer_norm=layer_norm)
        else:
            print('Unexpected rnn type')
            exit()
    
    def forward(self, input, h, h2=None):
        output, hidden = self.rnn(input, h, h2)
        return output, hidden

    def init_hidden(self, batch_size):
        bidirectional = 2 if self.bidirectional else 1
        h = Variable(torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size))

        if self.rnn_type == 'gru':
            return h.cuda() if use_cuda else h
        else:
            c = Variable(torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size))
            return (h.cuda(), c.cuda()) if use_cuda else (h, c)

class MDRNN(nn.Module):
    def __init__(self, 
                 input_size=16,
                 hidden_size=128,
                 output_size=10,
                 layer_norm=False,
                 axis=4,
                 rnn_type='gru'
                 ):
        super(MDRNN, self).__init__()
        self.rnn_type = rnn_type
        rnns = []
        for _ in range(axis):
            rnns.append(RNN(input_size, 
                            hidden_size,
                            layer_norm=layer_norm,
                            rnn_type=rnn_type))
        self.rnns = nn.ModuleList(rnns)
        self.output = nn.Linear(hidden_size * len(self.rnns), output_size)
    
    def forward(self, input):
        '''
        Args:
            input: (batch, n, n)
        '''
        batch = input.size(0)
        n = input.size(1)

        final_hidden = None
        # 2d case, we need general case?
        grid = 4
        x_ori, x_stop, x_steps = [0, 0, n-1, n-1], [n-(grid-1), n-(grid-1), (grid-1), (grid-1)], [1, 1, -1, -1]
        y_ori, y_stop, y_steps = [0, n-1, 0, n-1], [n-(grid-1), (grid-1), n-(grid-1), (grid-1)], [1, -1, 1, -1]
        for axis_idx, rnn in enumerate(self.rnns):
            last_row = []
            for i in range(y_ori[axis_idx], y_stop[axis_idx], y_steps[axis_idx]):
                row = []
                last_h = None
                for idx, j in enumerate(range(x_ori[axis_idx], x_stop[axis_idx], x_steps[axis_idx])):
                    # handle hidden from last row
                    if len(last_row) == 0:
                        h = rnn.init_hidden(batch)
                    else:
                        h = last_row[idx]
                    # handle hidden from last column
                    if last_h is None:
                        h2 = rnn.init_hidden(batch)
                    else:
                        h2 = last_h

                    # handle input grid
                    if y_steps[axis_idx] > 0:
                        i_start, i_end = i, i + grid
                    else:
                        i_start, i_end = i - grid, i
                    if x_steps[axis_idx] > 0:
                        j_start, j_end = j, j + grid
                    else:
                        j_start, j_end = j - grid, j

                    input_step = input[:, i_start:i_end, j_start:j_end].contiguous()
                    _, last_h = rnn(input_step.view(batch, -1).unsqueeze(0), h, h2)
                    row.append(last_h)
                last_row = row
            
            if self.rnn_type == 'lstm':
                output_hidden = row[-1][0]
            else:
                output_hidden = row[-1]

            if final_hidden is None:
                final_hidden = output_hidden.squeeze(0)
            else:
                final_hidden = torch.cat((final_hidden, output_hidden.squeeze(0)), 1)
        return  F.log_softmax(self.output(final_hidden), dim=1)


class MDRNN2D(nn.Module):
    """
    Implementation of a 2D multi directionnal recurrent neural network
    """
    def __init__(self, input_size, hidden_size, no_cuda=False, bloc_size=(1, 1), layer_norm=False, rnn_type='gru'):
        super(MDRNN2D, self).__init__()
        
        self.bloc_x = bloc_size[0]
        self.bloc_y = bloc_size[1]
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.no_cuda = no_cuda
        
        # Creating a MDRNN for each 4 possible directions
        rnns = []
        for _ in range(4):
            rnns.append(RNN(input_size * self.bloc_x * self.bloc_y, 
                            hidden_size,
                            layer_norm=layer_norm,
                            rnn_type=rnn_type))
        self.rnns = nn.ModuleList(rnns)
    
    def forward(self, input):
        """
        Args:
            input: (batch size, input size, height, width)
        Outputs:
            output: (batch size, hidden size, directions, new_height, new_width)
        """
        batch, input_size, height, width = input.size()
        directions = 4
        
        # ~ output_map = torch.ones(batch, self.hidden_size, directions, height, width)
        output_map_ = torch.ones(directions, height, width, batch, self.hidden_size)
        if not self.no_cuda:
            output_map_ = output_map_.cuda()
    
        
        final_hidden = None
        # 2d case, we need general case?
        x_ori, x_stop, x_steps = [0, 0, width-1, width-1], [width-1, width-1, 0, 0], [1, 1, -1, -1]
        y_ori, y_stop, y_steps = [0, height-1, 0, height-1], [height-1, 0, height-1, 0], [1, -1, 1, -1]
        for axis_idx, rnn in enumerate(self.rnns):
            last_row = []
            for i in range(y_ori[axis_idx], y_stop[axis_idx], y_steps[axis_idx]):
                row = []
                last_h = None
                for idx, j in enumerate(range(x_ori[axis_idx], x_stop[axis_idx], x_steps[axis_idx])):
                    # handle hidden from last row
                    if len(last_row) == 0:
                        h = rnn.init_hidden(batch)
                    else:
                        h = last_row[idx]
                    # handle hidden from last column
                    if last_h is None:
                        h2 = rnn.init_hidden(batch)
                    else:
                        h2 = last_h

                    # handle input grid
                    if y_steps[axis_idx] > 0:
                        i_start, i_end = i, i + 1
                    else:
                        i_start, i_end = i - 1, i
                    if x_steps[axis_idx] > 0:
                        j_start, j_end = j, j + 1
                    else:
                        j_start, j_end = j - 1, j

                    input_step = input[:, :, i_start:i_end, j_start:j_end].contiguous()
                    output, last_h = rnn(input_step.view(batch, -1).unsqueeze(0), h, h2)
                    row.append(last_h)
                    
                    # Add the output to the current output map
                    """
                    for b in range(batch):
                        for h in range(self.hidden_size):
                            output_map[b][h][axis_idx][i][j] = output[0][b][h]
                    """
                    output_map_[axis_idx][i][j] = output[0]
                            
                last_row = row
            
            if self.rnn_type == 'lstm':
                output_hidden = row[-1][0]
            else:
                output_hidden = row[-1]

            if final_hidden is None:
                final_hidden = output_hidden.squeeze(0)
            else:
                final_hidden = torch.cat((final_hidden, output_hidden.squeeze(0)), 1)
        
        output_map = output_map_.permute(3, 4, 0, 1, 2)
        return output_map
