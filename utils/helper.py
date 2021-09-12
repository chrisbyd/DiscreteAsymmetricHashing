import torch
import numpy as np
from torch.autograd import Variable



def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 20, 30, 40, 50]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10




def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S


def calc_loss(V, U, S, code_length, select_index, gamma, opt):
    num_database = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length*S) ** 2
    V_omega = V[select_index, :]
    quantization_loss = (U-V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)
    return loss


def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B
