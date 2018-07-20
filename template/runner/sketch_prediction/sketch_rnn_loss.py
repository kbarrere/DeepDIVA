import torch
import math
import torch.nn as nn

class SketchRnnLoss(nn.Module):
    """
    TODO: Something
    """

    def __init__(self, no_cuda = True, conditional=False):
        super(SketchRnnLoss, self).__init__()

        self.no_cuda = no_cuda
        self.conditional = conditional

        return

    def forward(self, input, target, mu, presig, wkl):
        """
        Computes the loss of the model following the Sketch Rnn one

        :param input:
            The output of the network.
        :param target:
            What is the target to fit to.
            Usually it is the input sketch
        :param mu:
            Generated mean of the latent vector z by the encoder
        :param presig:
            Generated standard deviation of the latent vector z by the encoder
        :param wkl:
            Weigth of the Lkl term (Loss = Lr + wkl * Lkl)
        :return:
            Loss : The computed loss
        """
        # Start by computing some usefull variables

        input = input.permute(1, 0, 2)

        batch_size = len(input[0])
        Nmax = len(input)
        Nz = len(mu[0])
        eps = 1e-9 # To avoid log(0)
        M = len(input[0][0]) // 6

        # Computes Ns:
        target_abs = torch.abs(target)
        target_sum = torch.sum(target_abs[:, :, :4], dim=2)
        target_is_zero = torch.eq(target_sum, 0)
        target_is_one = torch.eq(target[:, :, 4], 1)
        is_finished = target_is_one * target_is_zero
        is_not_finished = 1 - is_finished
        if self.no_cuda:
            is_not_finished = is_not_finished.type(torch.FloatTensor)
        else:
            is_not_finished = is_not_finished.type(torch.cuda.FloatTensor)

        ind_x1_ = [0]
        ind_x2_ = [1]
        ind_pk_ = [2, 3, 4]

        ind_pi_ = range(0, 20)
        ind_mean1_ = range(20, 40)
        ind_mean2_ = range(40, 60)
        ind_std1_ = range(60, 80)
        ind_std2_ = range(80, 100)
        ind_cor_ = range(100, 120)
        ind_qk_ = range(120, len(input[0][0]))

        if self.no_cuda:
            ind_x1 = torch.LongTensor(ind_x1_)
            ind_x2 = torch.LongTensor(ind_x2_)
            ind_pk = torch.LongTensor(ind_pk_)

            ind_pi = torch.LongTensor(ind_pi_)
            ind_mean1 = torch.LongTensor(ind_mean1_)
            ind_mean2 = torch.LongTensor(ind_mean2_)
            ind_std1 = torch.LongTensor(ind_std1_)
            ind_std2 = torch.LongTensor(ind_std2_)
            ind_cor = torch.LongTensor(ind_cor_)
            ind_qk = torch.LongTensor(ind_qk_)
        else:
            ind_x1 = torch.cuda.LongTensor(ind_x1_)
            ind_x2 = torch.cuda.LongTensor(ind_x2_)
            ind_pk = torch.cuda.LongTensor(ind_pk_)

            ind_pi = torch.cuda.LongTensor(ind_pi_)
            ind_mean1 = torch.cuda.LongTensor(ind_mean1_)
            ind_mean2 = torch.cuda.LongTensor(ind_mean2_)
            ind_std1 = torch.cuda.LongTensor(ind_std1_)
            ind_std2 = torch.cuda.LongTensor(ind_std2_)
            ind_cor = torch.cuda.LongTensor(ind_cor_)
            ind_qk = torch.cuda.LongTensor(ind_qk_)

        t_x1_ = torch.index_select(target, 2, ind_x1)
        t_x2_ = torch.index_select(target, 2, ind_x2)
        t_pk_ = torch.index_select(target, 2, ind_pk)

        t_x1 = t_x1_.permute(1, 0, 2)
        t_x2 = t_x2_.permute(1, 0, 2)
        t_pk = t_pk_.permute(1, 0, 2)

        t_pi = torch.index_select(input, 2, ind_pi)
        t_mean1 = torch.index_select(input, 2, ind_mean1)
        t_mean2 = torch.index_select(input, 2, ind_mean2)
        t_std1 = torch.index_select(input, 2, ind_std1)
        t_std2 = torch.index_select(input, 2, ind_std2)
        t_cor = torch.index_select(input, 2, ind_cor)
        t_qk = torch.index_select(input, 2, ind_qk)

        if self.no_cuda:
            t_x1 = t_x1.type(torch.FloatTensor)
            t_x2 = t_x2.type(torch.FloatTensor)
            t_pk = t_pk.type(torch.FloatTensor)
        else:
            t_x1 = t_x1.type(torch.cuda.FloatTensor)
            t_x2 = t_x2.type(torch.cuda.FloatTensor)
            t_pk = t_pk.type(torch.cuda.FloatTensor)

        t_norm1 = t_x1 - t_mean1
        t_norm2 = t_x2 - t_mean2

        t_s1s2 = t_std1 * t_std2

        t_z = ((t_norm1**2)/(t_std1**2)) + ((t_norm2**2)/(t_std2**2)) - ((2 * t_cor * t_norm1 * t_norm2)/t_s1s2)

        t_neg_rho = 1 - t_cor**2

        t_num = torch.exp(- 1 * t_z / (2 * t_neg_rho))
        t_denom = 2 * math.pi * t_s1s2 * torch.sqrt(t_neg_rho)
        t_n = t_num / t_denom

        t_mul = t_pi * t_n
        t_sum = torch.sum(t_mul, dim=2)

        t_log = torch.log(t_sum + eps)
        # TODO: multiply by a vector of 1 and zero to keep the terms before Ns
        is_not_finished = is_not_finished.permute(1, 0)
        t_log_below_ns = t_log * is_not_finished
        t_sum_log = -1 * torch.sum(t_log_below_ns)
        Ls = t_sum_log / (Nmax * batch_size)

        t_log_qk = torch.log(t_qk + eps)
        t_pk_log_qk = t_pk * t_log_qk
        t_sum_pk_qk = torch.sum(t_pk_log_qk)

        Lp = -1 * t_sum_pk_qk / (Nmax * batch_size)

        Lr = Ls + Lp

        Lkl = torch.FloatTensor([0])
        if self.conditional:
            t_lkl_num = 1 + presig - mu**2 - torch.exp(presig)
            t_lkl_sum = torch.sum(t_lkl_num)

            Lkl = t_lkl_sum * (-1 / (2 * Nz * batch_size))
        if self.no_cuda:
            min_value = torch.FloatTensor([0.2])
        else:
            min_value = torch.cuda.FloatTensor([0.2])
        Lkl = torch.max(Lkl, min_value)

        Loss = Lr + wkl * Lkl

        """
        print("resulting loss :")
        print("----------------")
        print(Ls)
        print(Lp)
        print(Lr)
        print(Lkl)
        print(Loss)
        """

        return Loss, Lr, Lkl, Ls, Lp
