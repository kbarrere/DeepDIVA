import torch
import math
import torch.nn as nn

class SketchRnnLoss(nn.Module):
    """
    TODO: Something
    """

    def __init__(self, no_cuda = True):
        super(SketchRnnLoss, self).__init__()

        self.no_cuda = no_cuda

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
        batch_size = len(input)
        Nmax = len(input[0])
        Ns = len(target[0]) #TODO:change it ! Ns should be a vector and computed as the first value wherer input[batch][seq] ???
        Nz = len(mu[0])
        eps = 1e-6 # To avoid log(0)
        M = len(input[0][0]) // 6


        # Computes Lr first
        Ls = 0
        Lp = 0
        for b in range(batch_size):
            for i in range(Nmax):
                yi = input[b][i]

                if i < Ns:
                    # Ls is computed only for the first Ns point
                    x1 = target[b][i][0] # x shift of the input sequence/target
                    x2 = target[b][i][1]  # y shift of the input sequence/target
                    if self.no_cuda:
                        x1 = x1.type(torch.FloatTensor)
                        x2 = x2.type(torch.FloatTensor)
                    else:
                        x1 = x1.type(torch.cuda.FloatTensor)
                        x2 = x2.type(torch.cuda.FloatTensor)

                    sum = 0
                    for j in range(M):
                        pi = yi[6 * j]
                        mean1 = yi[6 * j + 1]
                        mean2 = yi[6 * j + 2]
                        std1 = yi[6 * j + 3]
                        std2 = yi[6 * j + 4]
                        cor = yi[6 * j + 5]
                        # Equation (24) and (25) in http://arxiv.org/abs/1308.0850
                        norm1 = x1 - mean1
                        norm2 = x2 - mean2
                        Z = ((norm1 ** 2) / (std1 ** 2)) + ((norm2 ** 2) / (std2 ** 2)) - (2 * cor * norm1 * norm2) / (std1 * std2)
                        #TODO: problem with division by 0
                        N = torch.exp( -1 * Z / (2 * (1 - cor ** 2))) / (2 * math.pi * std1 * std2 * torch.sqrt(1 - cor ** 2))
                        #print(N)
                        sum += pi * N
                        #print(sum)

                    Ls += torch.log(sum + eps)

                # Computes Lp
                q1 = yi[-3]
                q2 = yi[-2]
                q3 = yi[-1]

                if self.no_cuda:
                    q1 = q1.type(torch.FloatTensor)
                    q2 = q2.type(torch.FloatTensor)
                    q3 = q3.type(torch.FloatTensor)
                else:
                    q1 = q1.type(torch.cuda.FloatTensor)
                    q2 = q2.type(torch.cuda.FloatTensor)
                    q3 = q3.type(torch.cuda.FloatTensor)

                #default case when i >= Ns
                p1, p2, p3 = 0, 0, 1
                if i < Ns:
                    p1 = target[b][i][2]

                    p2 = target[b][i][3]

                    p3 = target[b][i][4]


                    if self.no_cuda:
                        p1 = p1.type(torch.FloatTensor)
                        p2 = p2.type(torch.FloatTensor)
                        p3 = p3.type(torch.FloatTensor)
                    else:
                        p1 = p1.type(torch.cuda.FloatTensor)
                        p2 = p2.type(torch.cuda.FloatTensor)
                        p3 = p3.type(torch.cuda.FloatTensor)



                Lp += p1 * torch.log(q1 + eps) + p2 * torch.log(q2 + eps) + p3 * torch.log(q3 + eps)

        Lr = Ls + Lp
        Lr *= (-1 / (Nmax * batch_size))

        # Then computes Lkl
        Lkl = 0
        for b in range(batch_size):
            for i in range(Nz):
                Lkl += 1 + presig[b][i] - mu[b][i] ** 2 - torch.exp(presig[b][i])
        Lkl *= (-1 / (2 * Nz * batch_size))

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

        return Loss
