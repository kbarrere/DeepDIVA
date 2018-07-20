import torch
import torch.nn as nn

class Sketch_RNN(nn.Module):
    """
    TODO : good introduction with citations probably
    
    Attributes
    ----------
    TODO : attributes
    """

    #TODO parameters
    def __init__(self, no_cuda=True, conditional=False):
        """
        TODO
        """

        super(Sketch_RNN, self).__init__()

        self.no_cuda = no_cuda
        self.conditional = conditional
        print("HELLO?")

        sketch_point_size = 5
        max_seq_len = 256

        if self.conditional:
            enc_input_size = sketch_point_size
            enc_hidden_size = max_seq_len
            enc_num_layers = 1
            enc_bias = True
            enc_batch_first = True
            enc_dropout = 0
            enc_bidirectional = True

        self.expected_input_size = (max_seq_len, sketch_point_size)

        #TODO change the RNN type based on parameters ?
        #Encoder : Bidirectionnal Recurrent Neural Network
        if self.conditional:
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

        if self.conditional:
            #Fully connected layer
            self.fc_sigma = nn.Linear(self.h_size, self.z_size)
            self.fc_mu = nn.Linear(self.h_size, self.z_size)

        dec_input_size = sketch_point_size + self.z_size

        if self.conditional:
            self.dec_hidden_size = 512
        else:
            self.dec_hidden_size = 1024
        dec_bias = True

        self.fc_h0 = nn.Sequential(
            nn.Linear(self.z_size, self.dec_hidden_size),
            nn.Tanh()
        )

        self.num_mixture = 20

        #Creates a single LSTM cell and iterates over it step by step in the forward function
        self.dec_rnn = nn.modules.rnn.LSTMCell(
            dec_input_size,
            self.dec_hidden_size,
            dec_bias
        )

        #at the end of each step of the decoder, takes the hidden state and pass it through this fully connected network
        self.fc_dec = nn.Linear(self.dec_hidden_size, 6*self.num_mixture + 3)

    def forward(self, s):
        """
        Computes forward pass on the network
        TODO
        the shape of the input sequence 's' should be
        (batch, seq_len, input_size)
        """

        n_max = 256
        batch_size = len(s)
        n_s = []
        for b in range(batch_size):
            n_s.append(len(s[b]))

        if self.no_cuda:
            s = s.type(torch.FloatTensor)
        else:
            s = s.type(torch.cuda.FloatTensor)

        presig = torch.zeros(self.z_size)
        mu = torch.zeros(self.z_size)

        if not self.no_cuda:
            presig = presig.type(torch.cuda.FloatTensor)
            mu = mu.type(torch.cuda.FloatTensor)

        if self.conditional:
            h, _ = self.enc_brnn(s)
            h = h[:,-1] #only takes the results of the last point of the sequence

            presig = self.fc_sigma(h)
            mu = self.fc_mu(h)

            if not self.no_cuda:
                mu = mu.type(torch.cuda.FloatTensor)

            sigma = torch.exp(torch.div(presig, 2))

            if not self.no_cuda:
                sigma = sigma.type(torch.cuda.FloatTensor)

        
            #generate a vector epsilon of size z_size, with each value following N(0,1)
            epsilon = torch.randn(batch_size, self.z_size)

            if not self.no_cuda:
                epsilon = epsilon.type(torch.cuda.FloatTensor)

            z = torch.add(mu, torch.mul(sigma, epsilon))
        else:
            z = torch.randn(batch_size, self.z_size)

        if not self.no_cuda:
            z = z.type(torch.cuda.FloatTensor)

        h_i = self.fc_h0(z)

        #decoder rnn with gaussian mixture model

        # first iteration
        tmp_s = []
        tmp_c = []
        for i in range(batch_size):
            tmp_s.append([0., 0., 1., 0., 0.])
            tmp_c.append([0] * self.dec_hidden_size)

        output = torch.ones(n_max, batch_size, 6 * self.num_mixture + 3)

        if not self.no_cuda:
            output = output.type(torch.cuda.FloatTensor)

        if self.no_cuda:
            s_0 = torch.FloatTensor(tmp_s)
            c_i = torch.FloatTensor(tmp_c)
        else:
            s_0 = torch.cuda.FloatTensor(tmp_s)
            c_i = torch.cuda.FloatTensor(tmp_c)

        for i in range(n_max):

            if i == 0:
                s_i = s_0
            else:
                s_i = s[:, i-1]

            x_i = torch.cat((s_i, z), 1)

            h_i, c_i = self.dec_rnn(x_i, (h_i, c_i))

            y_i = self.fc_dec(h_i)

            ind_pi_ = []
            ind_mean1_ = []
            ind_mean2_ = []
            ind_std_x_ = []
            ind_std_y_ = []
            ind_cor_ = []
            for j in range(self.num_mixture):
                ind_pi_.append(6*j)
                ind_mean1_.append(6 * j + 1)
                ind_mean2_.append(6 * j + 2)
                ind_std_x_.append(6*j + 3)
                ind_std_y_.append(6*j + 4)
                ind_cor_.append(6*j + 5)
            ind_qk_ = [6*self.num_mixture, 6*self.num_mixture+1, 6*self.num_mixture+2]

            if self.no_cuda:
                ind_pi = torch.LongTensor(ind_pi_)
                ind_mean1 = torch.LongTensor(ind_mean1_)
                ind_mean2 = torch.LongTensor(ind_mean2_)
                ind_std_x = torch.LongTensor(ind_std_x_)
                ind_std_y = torch.LongTensor(ind_std_y_)
                ind_cor = torch.LongTensor(ind_cor_)
                ind_qk = torch.LongTensor(ind_qk_)
            else:
                ind_pi = torch.cuda.LongTensor(ind_pi_)
                ind_mean1 = torch.cuda.LongTensor(ind_mean1_)
                ind_mean2 = torch.cuda.LongTensor(ind_mean2_)
                ind_std_x = torch.cuda.LongTensor(ind_std_x_)
                ind_std_y = torch.cuda.LongTensor(ind_std_y_)
                ind_cor = torch.cuda.LongTensor(ind_cor_)
                ind_qk = torch.cuda.LongTensor(ind_qk_)

            t_pi_ = torch.index_select(y_i, 1, ind_pi)
            t_mean1 = torch.index_select(y_i, 1, ind_mean1)
            t_mean2 = torch.index_select(y_i, 1, ind_mean2)
            t_std_x_ = torch.index_select(y_i, 1, ind_std_x)
            t_std_y_ = torch.index_select(y_i, 1, ind_std_y)
            t_cor_ = torch.index_select(y_i, 1, ind_cor)
            t_qk_ = torch.index_select(y_i, 1, ind_qk)

            exp_pi = torch.exp(t_pi_)
            exp_sum_pi = torch.sum(exp_pi, dim=1)
            exp_pi_p = exp_pi.permute(1, 0)
            t_pi = torch.div(exp_pi_p, exp_sum_pi)
            t_pi = t_pi.permute(1, 0)

            t_std_x = torch.exp(t_std_x_)
            t_std_y = torch.exp(t_std_y_)

            t_cor = torch.tanh(t_cor_)

            exp_qk = torch.exp(t_qk_)
            exp_sum_qk = torch.sum(exp_qk)
            t_qk = exp_qk / exp_sum_qk

            t_cat1 = torch.cat((t_pi, t_mean1), dim=1)
            t_cat2 = torch.cat((t_mean2, t_std_x), dim=1)
            t_cat3 = torch.cat((t_std_y, t_cor), dim=1)
            t_cat4 = torch.cat((t_cat1, t_cat2), dim=1)
            t_cat5 = torch.cat((t_cat3, t_qk), dim=1)
            t_cat = torch.cat((t_cat4, t_cat5), dim=1)

            #TODO
            output[i] = t_cat

        output = output.permute(1, 0, 2)

        return output, mu, presig
