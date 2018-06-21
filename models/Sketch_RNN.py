import torch.nn as nn


class Sketch_RNN(nn.Module):
    """
    TODO : good introduction with citations probably
    
    Attributes
    ----------
    TODO : attributes
    """

    #TODO parameters
    def __init__(self):
        """
        TODO
        """

        super(Sketch_RNN, self).__init__()

        enc_input_size = 5
        enc_hidden_size = 256
        enc_num_layers = 1
        enc_bias = True
        enc_batch_first = False
        enc_dropout = 0
        enc_bidirectional = True

        #TODO change the RNN type based on parameters ?
        #Encoder : Bidirectionnal Recurrent Neural Network
        self.enc_brnn = nn.modules.rnn.LSTM(
            enc_input_size,
            enc_hidden_size,
            num_layers = enc_num_layers,
            bias = enc_bias,
            batch_first = enc_batch_first,
            dropout = enc_dropout,
            bidirectional = enc_bidirectional
        )

        #h is the resulting vector from the concatenation of the outputs of the encoder
        self.h_size = 2 * enc_hidden_size
        #z is a latent vector that is randomly encoded and conditionned by the input
        self.z_size = 128

        #Fully connected layer
        self.fc_sigma = nn.Linear(self.h_size, self.z_size)
        self.fc_mu = nn.Linear(self.h_size, self.Z_size)


        dec_hidden_size = 512

        self.fc_h0 = nn.Sequential(
            nn.Linear(self.z_size, dec_hidden_size),
            nn.Tanh()
        )

    def forward(self, s):
        """
        Computes forward pass on the netwok
        TODO
        """

        h = self.enc_brnn(s)

        sigma_ = self.fc_sigma(h)
        mu = self.fc_mu(h)
        sigma = torch.exp(torch.div(sigma_,2))
        
        #generate a vector epsilon of size z_size, with each value following N(0,1)
        epsilon = torch.randn(self.z_size)
        z = torch.add(mu + torch.mul(sigma * epsilon))
        h0 = self.fc_h0(z)

        return something
