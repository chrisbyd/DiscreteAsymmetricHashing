import torch 
import numpy as np
from dataset import get_loaders
from utils.subset_sampler import SubsetSampler
from utils.helper import adjusting_learning_rate
from utils.helper import calc_loss, calc_sim, encode
import utils.calc_hr as calc_hr
import logging
import time
import utils.subset_sampler as subsetsampler
from utils.adsh_loss import ADSHLoss
import torch.optim as optim
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils.cnn_model as cnn_model
from validate import validate
from utils.discrete_loss import solve_dcc
"""
The training function for this asymmetric hashing function
"""

def train(opt,code_length):
    logger_name = "{}_{}_{}".format(opt.exp_name, opt.dataset_name, str(code_length))
    logger = logging.getLogger(logger_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda:%d" % int(opt.gpu))


    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.train_batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_samples = opt.num_samples
    gamma = opt.gamma



    '''
    dataset preprocessing
    '''
  
    dset_database, dset_test, database_labels, test_labels = get_loaders(opt)
    num_database, num_test = database_labels.shape[0], test_labels.shape[0]
    print("The database has {} points, the test has {} points".format(num_database,num_test))
    
  

    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_length)
    model.cuda()
    adsh_loss = ADSHLoss(gamma, code_length, num_database)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    V = np.zeros((num_database, code_length))

    model.train()
    best_map = 0
    for iter in range(max_iter):
        iter_time = time.time()
        # '''
        # sampling and construct similarity matrix
        # '''
        select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=batch_size,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=4)
        # '''
        # learning deep neural network: feature learning
        # '''

        sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, database_labels)
        U = np.zeros((num_samples, code_length), dtype=np.float)
        
        N = num_samples
        UT = torch.zeros(code_length, N).to(device)
        B = torch.randn(code_length, N).sign().to(device)
        Y = sample_label.t().type(torch.FloatTensor).to(device)
        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                batch_size_ = train_label.size(0)
                u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration+1)*batch_size)) - 1, batch_size_, dtype=int)
                train_input = Variable(train_input.cuda())

                output = model(train_input)
                S = Sim.index_select(0, torch.from_numpy(u_ind))
                U[u_ind, :] = output.cpu().data.numpy()
                UT[:, u_ind] = output.t().data 

                model.zero_grad()
                loss = adsh_loss(output, V, S, B.t()[u_ind,:])
                logger.info("Iter:{}, iteration {}, the asymmetric loss is {}".format(iter, iteration,loss))
                loss.backward()
                optimizer.step()
        adjusting_learning_rate(optimizer, iter)

        '''
        learning binary codes: discrete coding
        '''
        barU = np.zeros((num_database, code_length))
        barU[select_index, :] = U
        Q = -2*code_length*Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * barU
        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
        iter_time = time.time() - iter_time


        W = torch.inverse(B @ B.t() + 1.0 / 1e-2 * torch.eye(code_length, device= device)) @ B @ Y.t()
     
        B = solve_dcc(W, Y, UT, B)

        

        # loss_ = calc_loss(V, U, Sim.cpu().numpy(), code_length, select_index, gamma, opt)
        # logger.info('[Iteration: %3d/%3d][Train Loss: %.4f]', iter, max_iter, loss_)
        
        if iter !=0 and (iter + 1) % opt.test_interval == 0:
            curr_map = validate(opt, code_length, iter, best_map, net= model, gallery_codes= V)
            if curr_map > best_map:
                best_map = curr_map
            logger.info("The map is {}".format(curr_map))



    '''
    training procedure finishes, evaluation
    '''
    model.eval()
    validate(opt, code_length, iter, best_map, net= model, gallery_codes= V)
    

    
