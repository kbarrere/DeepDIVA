import torch
import torch.nn as nn
import numpy as np
import random


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

        sketch_point_size = 5

        enc_input_size = sketch_point_size
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

        dec_input_size = sketch_point_size
        self.dec_hidden_size = 512
        dec_bias =  True

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

        #at the end of each step of the decoder, takes the hiddeb state and pass it through this fully connected network
        self.fc_dec = nn.Linear(self.dec_hidden_size, 6*self.num_mixture + 3)

    def forward(self, s):
        """
        Computes forward pass on the network
        TODO
        the shape of the input sequence 's' should be
        (seq_len, batch, input_size)
        """

        n_max = 256

        h = self.enc_brnn(s)

        sigma_ = self.fc_sigma(h)
        mu = self.fc_mu(h)
        sigma = torch.exp(torch.div(sigma_, 2))
        
        #generate a vector epsilon of size z_size, with each value following N(0,1)
        epsilon = torch.randn(self.z_size)
        z = torch.add(mu + torch.mul(sigma * epsilon))
        h_0 = self.fc_h0(z)

        #decoder rnn with gaussian mixture model

        #first iteration
        s_0 = torch.tensor([0., 0., 1., 0., 0.])
        x_0 = torch.cat(s_0, z)
        h_i, c_i = self.dec_rnn(x_0, h_0)

        y = self.fc_dec(h_i)

        # GMM

        # compute dx and dy for the predicted next point
        dx = 0
        dy = 0

        exp_sum_pi = torch.tensor([0.])
        for j in range(self.num_mixture):
            exp_sum_pi = torch.add(exp_sum_pi, torch.exp(y[6 * j]))

        # each value of y_i encode a parameter for the GMM
        for j in range(self.num_mixture):
            pi_ = y[6 * j]
            mean_x = y[6 * j + 1]
            mean_y = y[6 * j + 2]
            std_x_ = y[6 * j + 3]
            std_y_ = y[6 * j + 4]
            cor_ = y[6 * j + 5]

            std_x = torch.exp(std_x_)
            std_y = torch.exp(std_y_)

            pi = torch.div(torch.exp(pi_), exp_sum_pi)

            cor = torch.tanh(cor_)

            mean = [mean_x, mean_y]
            cov = [[std_x ** 2, cor], [cor, std_y ** 2]]

            # generate one random point (dx_j, dy_j) from a bivariate normal distribution
            x_array, y_array = np.random.multivariate_normal(mean, cov, 1).T

            dx = dx + pi * x_array[0]
            dy = dy + pi * y_array[0]

        # generate p1, p2 and p3
        q1_ = y[-3]
        q2_ = y[-2]
        q3_ = y[-1]

        expq1 = torch.exp(q1_)
        expq2 = torch.exp(q2_)
        expq3 = torch.exp(q3_)
        exp_sum_q = expq1 + expq2 + expq3

        q1_ = q1_ / exp_sum_q
        q2_ = q2_ / exp_sum_q
        q3_ = q3_ / exp_sum_q

        sum = q1_ + q2_ + q3_

        q1 = q1_ / sum
        q2 = q2_ / sum
        q3 = q3_ / sum

        # generate p1, p2 and p3 based on bernoulli distribution
        p1 = 0
        p2 = 0
        p3 = 0

        p = random.random()
        if p < q1:
            p1 = 1

        p = random.random()
        if p < q2:
            p2 = 1

        p = random.random()
        if p < q3:
            p3 = 1

        predicted_point = torch.tensor(dx, dy, p1, p2, p3)
        predicted_sketch = torch.tensor([predicted_point.numpy])

        h_i1 = h_i
        c_i1 = c_i

        for i in range(1, n_max):
            x_i = torch.cat(s_i1, z)
            h_i, c_i = self.dec_rnn(x_i, h_i1, c_i1)

            y_i = self.fc_dec(h_i)

            #GMM

            #compute dx and dy for the preticted next point
            dx = 0
            dy = 0

            exp_sum_pi = torch.tensor([0.])
            for j in range(self.num_mixture):
                exp_sum_pi = torch.add(exp_sum_pi, torch.exp(y_i[6*j]))

            #each value of y_i encode a parameter for the GMM
            for j in range(self.num_mixture):
                pi_ = y_i[6*j]
                mean_x = y_i[6*j + 1]
                mean_y = y_i[6*j + 2]
                std_x_ = y_i[6*j + 3]
                std_y_ = y_i[6*j + 4]
                cor_ = y_i[6*j + 5]

                std_x = torch.exp(std_x_)
                std_y = torch.exp(std_y_)

                pi = torch.div(torch.exp(pi_), exp_sum_pi)

                cor = torch.tanh(cor_)

                mean = [mean_x, mean_y]
                cov = [[std_x **2, cor], [cor, std_y **2]]

                #generate one random point (dx_j, dy_j) from a bivariate normal distribution
                x_array, y_array = np.random.multivariate_normal(mean, cov, 1).T

                dx = dx + pi * x_array[0]
                dy = dy + pi * y_array[0]

            #generate p1, p2 and p3
            q1_ = y_i[-3]
            q2_ = y_i[-2]
            q3_ = y_i[-1]

            expq1 = torch.exp(q1_)
            expq2 = torch.exp(q2_)
            expq3 = torch.exp(q3_)
            exp_sum_q = expq1 + expq2 + expq3

            q1_ = q1_ / exp_sum_q
            q2_ = q2_ / exp_sum_q
            q3_ = q3_ / exp_sum_q

            sum = q1_ + q2_ + q3_

            q1 = q1_ / sum
            q2 = q2_ / sum
            q3 = q3_ / sum

            #generate p1, p2 and p3 based on bernoulli distribution
            p1 = 0
            p2 = 0
            p3 = 0

            p = random.random()
            if p < q1:
                p1 = 1

            p = random.random()
            if p < q2:
                p2 = 1

            p = random.random()
            if p < q3:
                p3 = 1

            predicted_point = torch.tensor(dx, dy, p1, p2, p3)
            predicted_sketch = torch.cat(predicted_sketch, [predicted_point.numpy])

            h_i1 = h_i
            c_i1 = c_i

        # TODO: Change the output of the model according to the custom loss ?
        # TODO: At least create a function to get what is needed for the loss ?

        return predicted_sketch
