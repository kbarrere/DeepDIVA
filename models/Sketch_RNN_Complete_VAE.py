import torch
import torch.nn as nn
import numpy as np
import random
import logging
import sys


class Sketch_RNN_Complete_VAE(nn.Module):
    """
    TODO : good introduction with citations probably

    Attributes
    ----------
    TODO : attributes
    """

    # TODO parameters
    def __init__(self, no_cuda=True):
        """
        TODO
        """

        super(Sketch_RNN_Complete_VAE, self).__init__()

        self.no_cuda = no_cuda

        sketch_point_size = 5
        self.n_max = 256

        enc_input_size = sketch_point_size
        enc_hidden_size = 256
        enc_num_layers = 1
        enc_bias = True
        enc_batch_first = True
        enc_dropout = 0
        enc_bidirectional = True

        self.expected_input_size = (enc_hidden_size, sketch_point_size)

        # TODO change the RNN type based on parameters ?
        # Encoder : Bidirectionnal Recurrent Neural Network
        self.enc_brnn = nn.modules.rnn.LSTM(
            enc_input_size,
            enc_hidden_size,
            num_layers=enc_num_layers,
            bias=enc_bias,
            batch_first=enc_batch_first,
            dropout=enc_dropout,
            bidirectional=enc_bidirectional
        )

        # h is the resulting vector from the concatenation of the outputs of the encoder
        self.h_size = 2 * enc_hidden_size
        # z is a latent vector that is randomly encoded and conditionned by the input
        self.z_size = 128

        # Fully connected layer
        self.fc_sigma = nn.Linear(self.h_size, self.z_size)
        self.fc_mu = nn.Linear(self.h_size, self.z_size)

        dec_input_size = sketch_point_size + self.z_size
        self.dec_hidden_size = 512
        dec_bias = True

        self.fc_h0 = nn.Sequential(
            nn.Linear(self.z_size, self.dec_hidden_size),
            nn.Tanh()
        )

        self.num_mixture = 20

        # Creates a single LSTM cell and iterates over it step by step in the forward function
        self.dec_rnn = nn.modules.rnn.LSTMCell(
            dec_input_size,
            self.dec_hidden_size,
            dec_bias
        )

        # at the end of each step of the decoder, takes the hidden state and pass it through this fully connected network
        self.fc_dec = nn.Linear(self.dec_hidden_size, 6 * self.num_mixture + 3)

    def forward(self, s):
        """
        Computes forward pass on the network
        TODO
        the shape of the input sequence 's' should be
        (batch, seq_len, input_size)
        """

        sarray = [[
            [-12, 1, 1, 0, 0], [-6, 5, 1, 0, 0], [-29, 26, 1, 0, 0], [-21, 27, 1, 0, 0], [-6, 14, 1, 0, 0],
            [0, 25, 1, 0, 0], [4, 10, 1, 0, 0], [35, 31, 1, 0, 0], [28, 8, 1, 0, 0], [24, 0, 1, 0, 0],
            [19, -4, 1, 0, 0], [13, -7, 1, 0, 0], [15, -18, 1, 0, 0], [7, -13, 1, 0, 0], [4, -31, 1, 0, 0]
        ]]
        batch_size = len(sarray)

        n_s = []
        for b in range(batch_size):
            n_s.append(len(sarray[b]))
        n_s = torch.LongTensor(n_s)

        for b in range(batch_size):
            for j in range(self.n_max - n_s[b]):
                sarray[b].append([0., 0., 0., 0., 1.])

        s = torch.FloatTensor(sarray)
        if self.no_cuda:
            s = s.type(torch.FloatTensor)
        else:
            s = s.type(torch.cuda.FloatTensor)

        n_max = 256

        h, _ = self.enc_brnn(s)
        h = h[:, -1]  # only takes the results of the last point of the sequence

        presig = self.fc_sigma(h)
        mu = self.fc_mu(h)

        if not self.no_cuda:
            mu = mu.type(torch.cuda.FloatTensor)

        sigma = torch.exp(torch.div(presig, 2))

        if not self.no_cuda:
            sigma = sigma.type(torch.cuda.FloatTensor)

        # generate a vector epsilon of size z_size, with each value following N(0,1)
        epsilon = torch.randn(batch_size, self.z_size)

        if not self.no_cuda:
            epsilon = epsilon.type(torch.cuda.FloatTensor)

        z = torch.add(mu, torch.mul(sigma, epsilon))

        h_i = self.fc_h0(z)

        # decoder rnn with gaussian mixture model

        # first iteration
        tmp_s = []
        tmp_c = []
        for i in range(batch_size):
            tmp_s.append([0., 0., 1., 0., 0.])
            tmp_c.append([0] * self.dec_hidden_size)

        gen_points = [[0., 0., 1., 1., 1.]]

        if self.no_cuda:
            s_0 = torch.FloatTensor(tmp_s)
            c_i = torch.FloatTensor(tmp_c)
        else:
            s_0 = torch.cuda.FloatTensor(tmp_s)
            c_i = torch.cuda.FloatTensor(tmp_c)

        output = []

        for i in range(self.n_max):

            if i == 0:
                s_i = s_0
            else:
                s_i = s[:, i - 1]
                if i > torch.min(n_s):
                    # load the points gen_points
                    for b in range(batch_size):
                        if i > n_s[b]:
                            tmp = torch.LongTensor(gen_points[b])
                            s_i[b] = tmp

            x_i = torch.cat((s_i, z), 1)

            h_i, c_i = self.dec_rnn(x_i, (h_i, c_i))

            y_i = self.fc_dec(h_i)

            if True:
                ind_pi_ = []
                ind_mean1_ = []
                ind_mean2_ = []
                ind_std_x_ = []
                ind_std_y_ = []
                ind_cor_ = []
                for j in range(self.num_mixture):
                    ind_pi_.append(6 * j)
                    ind_mean1_.append(6 * j + 1)
                    ind_mean2_.append(6 * j + 2)
                    ind_std_x_.append(6 * j + 3)
                    ind_std_y_.append(6 * j + 4)
                    ind_cor_.append(6 * j + 5)
                ind_qk_ = [6 * self.num_mixture, 6 * self.num_mixture + 1, 6 * self.num_mixture + 2]

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

                # TODO: generates the points
                points = []
                for b in range(batch_size):
                    if i >= n_s[b]:
                        """
                        # Generates the points
                        dx = 0
                        dy = 0
                        for m in range(self.num_mixture):
                            mean_x = t_mean1[b][m].item()
                            mean_y = t_mean2[b][m].item()
                            std_x = t_std_x[b][m].item()
                            std_y = t_std_y[b][m].item()
                            cor = t_cor[b][m].item()
                            pi = t_pi[b][m].item()

                            mean = [mean_x, mean_y]
                            cov = [[std_x ** 2, cor * std_x * std_y], [cor * std_x * std_y, std_y ** 2]]

                            x, y = np.random.multivariate_normal(mean, cov, 1).T

                            dx += 1 * x
                            dy += 1 * y

                        q1 = t_qk[b][0]
                        q2 = t_qk[b][1]
                        q3 = t_qk[b][2]

                        print("-----------")
                        print(q1)
                        print(q2)
                        print(q3)

                        p1 = 0
                        p2 = 0
                        p3 = 0

                        r = random.random()
                        if r < q1:
                            p1 = 1
                        elif r < q1 + q2:
                            p2 = 1
                        else:
                            p3 = 1

                        points.append([int(dx), int(dy), p1, p2, p3])
                        """
                        # Generates the points
                        idx = get_pi_idx(random.random(), t_pi[b])
                        idx_eos = get_pi_idx(random.random(), t_qk[b])
                        eos = [0, 0, 0]
                        eos[idx_eos] = 1

                        dx, dy = sample_gaussian_2d(t_mean1[b][idx].item(), t_mean2[b][idx].item(),
                                                    t_std_x[b][idx].item(), t_std_y[b][idx].item(),
                                                    t_cor[b][idx].item())
                        points.append([int(dx), int(dy), eos[0], eos[1], eos[2]])

                    else:
                        points.append(sarray[b][i])

                gen_points = points

            output.append(gen_points[0])

        print('\n')
        print(output)
        np.save("pred", output)
        return output

def adjust_temp(pi_pdf, temp=1.0):
    pi_pdf = torch.log(pi_pdf) / temp
    pi_pdf = pi_pdf - torch.max(pi_pdf)
    pi_pdf = torch.exp(pi_pdf)
    pi_pdf = pi_pdf / torch.sum(pi_pdf)
    return pi_pdf

def get_pi_idx(x, pdf, temp=1.0):
    pdf = adjust_temp(pdf, temp)
    accumulate = 0.
    for i in range(len(pdf)):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    logging.error("Error with sampling ensemble.")
    sys.exit(-1)

def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0):
    mean = [mu1, mu2]
    temp2 = temp ** 2
    s1 = s1 * temp2
    s2 = s2 * temp2
    cov = [[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]]
    x, y = np.random.multivariate_normal(mean, cov, 1).T
    return x, y