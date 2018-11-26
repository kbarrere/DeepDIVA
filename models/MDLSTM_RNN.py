import logging

import torch.nn as nn


class BDLSTM(nn.Module):
	
	def __init__(self, input_size, output_size, dropout=0):
		logging.warning("BDSLM INIT TODO")
		
		super(BDLSTM, self).__init__()
		
		self.input_size = input_size
		self.output_size = output_size
		self.dropout = dropout
		
		self.lstm = nn.LSTM(self.input_size, self.output_size, batch_first=True, dropout=self.dropout)
		
	def forward(self, x):
		logging.warning("BDSLM FORWARD TODO")
		
		x = self.lstm(x)
		
		return x


class MDLSTM_RNN(nn.Module):
	
	def __init__(self, output_channels=80, expected_input_size=(128, 2174)):
		logging.warning("MDLSTM_RNN INIT TODO")
		
		super(MDLSTM_RNN, self).__init__()
		
		self.expected_input_size = expected_input_size
		
		self.bdlstm1 = BDLSTM(3, 4)
		self.conv1 = nn.Conv2d(4, 12, kernel_size=3, stride=1, padding=0),
			
		
	def forward(self, x):
		logging.warning("MDLSTM_RNN FORWARD TODO")
		logging.error(str(x.size()))
		x = self.bdlstm1(x)
		x = self.conv1(x)
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
