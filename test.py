import torch 
from validate import validate
from utils.logger import setup_logger
from config import get_parsed_args
import os.path as osp
import numpy as np
import utils.cnn_model as cnn_model


if __name__ == '__main__':
    opt = get_parsed_args()
   
    bits = [int(bit) for bit in opt.bits.split(',')]
    
    

    for bit in bits:
        name = "{}_{}_{}".format(opt.exp_name, opt.dataset_name, str(bit))
        logger = setup_logger(name, opt.log_dir, if_train= False)
        model = cnn_model.CNNNet(opt.arch, bit)
        model.cuda()
        save_model_path = opt.dataset_name + '-'+ opt.exp_name + '-' + str(bit) + '-model.pt'
        model_path = osp.join(opt.checkpoint_path,save_model_path)
        model.load_state_dict(torch.load(model_path))
        save_code_path = opt.dataset_name + '-'+ opt.exp_name + '-' + str(bit) + '-db_code.npy'
        db_code_path = osp.join(opt.checkpoint_path, 'db_code' ,save_code_path)
        database_code = np.load(db_code_path)
    

        mAP = validate(opt, bit, 0, 100, net= model, gallery_codes= database_code)
        logger.info('The overall map for this test is {}'.format(mAP))