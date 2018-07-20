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
    def __init__(self, no_cuda = True):
        """
        TODO
        """

        super(Sketch_RNN, self).__init__()

        self.no_cuda = no_cuda

        sketch_point_size = 5

        enc_input_size = sketch_point_size
        enc_hidden_size = 256
        enc_num_layers = 1
        enc_bias = True
        enc_batch_first = True
        enc_dropout = 0
        enc_bidirectional = True

        self.expected_input_size = (enc_hidden_size, sketch_point_size)

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
        self.fc_mu = nn.Linear(self.h_size, self.z_size)

        dec_input_size = sketch_point_size + self.z_size
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
        epsilon = torch.randn(self.z_size)

        if not self.no_cuda:
            epsilon = epsilon.type(torch.cuda.FloatTensor)

        z = torch.add(mu, torch.mul(sigma, epsilon))

        h_0 = self.fc_h0(z)

        #decoder rnn with gaussian mixture model

        #first iteration
        tmp_s = []
        tmp_c = []
        for i in range(batch_size):
            tmp_s.append([0., 0., 1., 0., 0.])
            tmp_c.append([0] * self.dec_hidden_size)

        if self.no_cuda:
            s_0 = torch.FloatTensor(tmp_s)
            c_0 = torch.FloatTensor(tmp_c)
        else:
            s_0 = torch.cuda.FloatTensor(tmp_s)
            c_0 = torch.cuda.FloatTensor(tmp_c)

        x_0 = torch.cat((s_0, z), 1)

        h_i, c_i = self.dec_rnn(x_0, (h_0, c_0))

        y = self.fc_dec(h_i)

        output = torch.ones(batch_size, n_max, 6 * self.num_mixture + 3)

        if not self.no_cuda:
            output = output.type(torch.cuda.FloatTensor)

        for b in range(batch_size):
            exp_sum_pi = torch.tensor([0.])
            if not self.no_cuda:
                exp_sum_pi = exp_sum_pi.type(torch.cuda.FloatTensor)
            for j in range(self.num_mixture):
                exp_sum_pi = torch.add(exp_sum_pi, torch.exp(y[b][6 * j]))

            # each value of y_i encode a parameter for the GMM
            for j in range(self.num_mixture):
                pi_ = y[b][6 * j]

                std_x_ = y[b][6 * j + 3]
                std_y_ = y[b][6 * j + 4]
                cor_ = y[b][6 * j + 5]

                std_x = torch.exp(std_x_)
                y[b][6 * j + 3] = std_x

                std_y = torch.exp(std_y_)
                y[b][6 * j + 4] = std_y

                pi = torch.div(torch.exp(pi_), exp_sum_pi)
                y[b][6 * j] = pi

                cor = torch.tanh(cor_)
                y[b][6 * j + 5] = cor

            q1_ = y[b][-3]
            q2_ = y[b][-2]
            q3_ = y[b][-1]

            expq1 = torch.exp(q1_)
            expq2 = torch.exp(q2_)
            expq3 = torch.exp(q3_)
            exp_sum_q = expq1 + expq2 + expq3

            q1 = expq1 / exp_sum_q
            q2 = expq2 / exp_sum_q
            q3 = expq3 / exp_sum_q

            y[b][-3] = q1
            y[b][-2] = q2
            y[b][-1] = q3

        #predicted_points = torch.tensor(y)
        #predicted_sketchs = torch.FloatTensor([y.numpy])

        #TODO:
        for b in range(batch_size):
            for i in range(6 * self.num_mixture + 3):
                output[b][0][i] = y[b][i]

        h_i1 = h_i
        c_i1 = c_i


        for i in range(1, n_max):
            # TODO: s_i1 = sequence
            # Default case when i >= Ns

            s_i1 = s[:, i-1]

            x_i = torch.cat((s_i1, z), 1)
            h_i, c_i = self.dec_rnn(x_i, (h_i1, c_i1))

            y_i = self.fc_dec(h_i)
            for b in range(batch_size):
                exp_sum_pi = torch.tensor([0.])
                if not self.no_cuda:
                    exp_sum_pi = exp_sum_pi.type(torch.cuda.FloatTensor)
                for j in range(self.num_mixture):
                    exp_sum_pi = torch.add(exp_sum_pi, torch.exp(y_i[b][6*j]))

                #each value of y_i encode a parameter for the GMM
                for j in range(self.num_mixture):
                    pi_ = y_i[b][6*j]
                    std_x_ = y_i[b][6*j + 3]
                    std_y_ = y_i[b][6*j + 4]
                    cor_ = y_i[b][6*j + 5]

                    std_x = torch.exp(std_x_)
                    y_i[b][6 * j + 3] = std_x
                    std_y = torch.exp(std_y_)
                    y_i[b][6 * j + 4] = std_y

                    pi = torch.div(torch.exp(pi_), exp_sum_pi)
                    y_i[b][6 * j] = pi

                    cor = torch.tanh(cor_)
                    y_i[b][6 * j + 5] = cor





                #generate p1, p2 and p3
                q1_ = y_i[b][-3]
                q2_ = y_i[b][-2]
                q3_ = y_i[b][-1]

                expq1 = torch.exp(q1_)
                expq2 = torch.exp(q2_)
                expq3 = torch.exp(q3_)
                exp_sum_q = expq1 + expq2 + expq3

                q1 = expq1 / exp_sum_q
                q2 = expq2 / exp_sum_q
                q3 = expq3 / exp_sum_q

                y_i[b][-3] = q1
                y_i[b][-2] = q2
                y_i[b][-1] = q3



            #predicted_points = torch.tensor(y_i)
            #predicted_sketchs = torch.cat(predicted_sketchs, [predicted_points.numpy])

               # TODO:
            for b in range(batch_size):
                for j in range(6 * self.num_mixture + 3):
                    output[b][i][j] = y_i[b][j]

            h_i1 = h_i
            c_i1 = c_i

        return output, mu, presig
