import torch.nn as nn
import torch
from torch.autograd import Variable

class ADSHLoss(nn.Module):
    def __init__(self, gamma, code_length, num_train):
        super(ADSHLoss, self).__init__()
        self.gamma = gamma
        self.code_length = code_length
        self.num_train = num_train

    def forward(self, u, V, S, B):
        batch_size = u.size(0)
        V = Variable(torch.from_numpy(V).type(torch.FloatTensor).cuda())
        S = Variable(S.cuda())
        square_loss = (u.mm(V.t())-self.code_length * S) ** 2
        quantization_loss = self.gamma * (B - u) ** 2
        loss = (square_loss.sum() + quantization_loss.sum()) /(self.num_train * batch_size)
        return loss
