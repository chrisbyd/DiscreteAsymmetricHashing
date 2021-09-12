
import numpy as np 
import torch
from train import train
from utils.logger import setup_logger
from config import get_parsed_args



if __name__=="__main__":
    global opt, logdir
    opt = get_parsed_args()
   
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        name = "{}_{}_{}".format(opt.exp_name, opt.dataset_name, str(bit))
        logger = setup_logger(name, opt.log_dir, if_train= True)
        train(opt, bit)
