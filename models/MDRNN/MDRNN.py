"""
Code from
https://github.com/exe1023/MDRNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.MDRNN.rnn import MDGRU, MDLSTM

class RNN(nn.Module):
    '''
    Wrapped RNN
    '''
    def __init__(self, input_size, hidden_size, layer_norm=False, dropout=0, rnn_type='gru', no_cuda=False):
        super(RNN, self).__init__()
        # these two paramters are now fixed (not implemented),
        # but we may need them?
        self.bidirectional = False
        self.num_layers = 1

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.no_cuda = no_cuda
        
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
            return h.cuda() if not self.no_cuda else h
        else:
            c = Variable(torch.zeros(bidirectional * self.num_layers, batch_size, self.hidden_size))
            return (h.cuda(), c.cuda()) if not self.no_cuda else (h, c)


class MDRNN2D(nn.Module):
    """
    Implementation of a 2D multi directionnal recurrent neural network
    """
    def __init__(self, input_size, hidden_size, no_cuda=False, bloc_size=(1, 1), dropout=0, layer_norm=False, rnn_type='gru'):
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
                            dropout=dropout,
                            rnn_type=rnn_type,
                            no_cuda=no_cuda))
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
        
        output_map = torch.ones(batch, self.hidden_size, directions, height, width)
        output_map_ = torch.ones(directions, height, width, batch, self.hidden_size)
        if not self.no_cuda:
            output_map_ = output_map_.cuda()
    
        final_hidden = None
        # 2d case, we need general case?
        x_ori, x_stop, x_steps = [0, 0, width-1, width-1], [width, width, -1, -1], [1, 1, -1, -1]
        y_ori, y_stop, y_steps = [0, height-1, 0, height-1], [height, -1, height, -1], [1, -1, 1, -1]

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
                        i_start, i_end = i, i + 1
                    if x_steps[axis_idx] > 0:
                        j_start, j_end = j, j + 1
                    else:
                        j_start, j_end = j, j + 1

                    input_step = input[:, :, i_start:i_end, j_start:j_end].contiguous()
                    output, last_h = rnn(input_step.view(batch, -1).unsqueeze(0), h, h2)
                    row.append(last_h)
                    
                    # Add the output to the current output map
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
        
        return final_hidden, output_map
