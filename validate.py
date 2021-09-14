import torch
import numpy as np
import utils.calc_hr as calc_hr
from utils.helper import encode
from utils.log_to_txt import results_to_txt
from dataset import get_loaders
import logging
import os
from torch.utils.data import DataLoader
from utils.tools import CalcTopMap

def validate(opt, bit_length,epoch_num, best_map, net = None, gallery_codes = None):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    # The test dataset info
    dset_database, dset_test, gallery_labels, query_labels = get_loaders(opt)
    num_database, num_test = gallery_labels.shape[0], query_labels.shape[0]
    print("The database has {} points, the test has {} points".format(num_database,num_test))
    '''
    model construction
    '''
    if net is None:
        raise NotImplementedError("This model is not implemented!")


    if gallery_codes is None:
        raise NotImplementedError("The database code must be provided!")
    model = net
    model.eval()
    testloader = DataLoader(dset_test, batch_size=1,
                            shuffle=False,
                            num_workers=4)
    qB = encode(model, testloader, num_test, bit_length)
    rB = gallery_codes
    mAP, cum_prec, cum_recall = CalcTopMap(qB, rB, query_labels.numpy(), gallery_labels.numpy(),
                               opt.topk)
    file_name = opt.machine_name +'_' + opt.dataset_name
    model_name = opt.exp_name + '_' + str(bit_length) + '_' + str(epoch_num)
    index_range = num_database // 100
    index = [i * 100 - 1 for i in range(1, index_range+1)]
    max_index = max(index)
    overflow = num_database - index_range * 100
    index = index + [max_index + i  for i in range(1,overflow + 1)]

    c_prec = cum_prec[index].tolist()
    c_recall = cum_recall[index].tolist()

    results_to_txt([mAP], filename=file_name, model_name=model_name, sheet_name='map')
    results_to_txt(c_prec, filename=file_name, model_name=model_name, sheet_name='prec_cum')
    results_to_txt(c_recall, filename=file_name, model_name=model_name, sheet_name='recall_cum')

    
    if mAP > best_map :

        if not os.path.exists(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        save_code_path = os.path.join(opt.checkpoint_path,'db_code')
        if not os.path.exists(save_code_path):
            os.makedirs(save_code_path)
     
        torch.save(net.state_dict(),
                    os.path.join(opt.checkpoint_path, opt.dataset_name + '-'+ opt.exp_name +"-" + str(mAP) +'-' + str(bit_length) + "-model.pt"))
        code_save_path = os.path.join(opt.checkpoint_path, 'db_code', opt.dataset_name + '-'+ opt.exp_name +"-" + str(mAP) +'-' + str(bit_length) + "-db_code.npy")
        np.save(code_save_path, gallery_codes)

    return mAP



