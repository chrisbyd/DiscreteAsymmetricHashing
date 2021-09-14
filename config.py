import argparse


def get_parsed_args():
    parser = argparse.ArgumentParser(description="ADSH demo")
    parser.add_argument('--bits', default='12,24,32,48', type=str,
                        help='binary code length (default: 12,24,32,48)')
    parser.add_argument('--gpu', default='0', type=str,
                        help='selected gpu (default: 1)')
    parser.add_argument('--arch', default='alexnet', type=str,
                        help='model name (default: resnet50)')
    parser.add_argument('--max-iter', default=60, type=int,
                        help='maximum iteration (default: 50)')
    parser.add_argument('--epochs', default=3, type=int,
                        help='number of epochs (default: 3)')
    parser.add_argument('--train-batch-size', default=64, type=int,
                        help='batch size (default: 64)')
    
    parser.add_argument('--test-batch-size', default=64, type=int,
                        help='batch size (default: 64)')

    parser.add_argument('--num-samples', default=2000, type=int,
                        help='hyper-parameter: number of samples (default: 2000)')
    parser.add_argument('--gamma', default=200, type=int,
                        help='hyper-parameter: gamma (default: 200)')
    parser.add_argument('--log-dir', default= './log', type= str ,
                        help= "the logging dir")
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='hyper-parameter: learning rate (default: 10**-3)')
    
    parser.add_argument('--dataset-name', default= 'cifar10', type= str, 
                        help= "Input the  training dataset name ")
    
    parser.add_argument('--machine-name', default= '1080', type= str, 
                        help= "Input the  training dataset name ")

    parser.add_argument('--checkpoint_path', default= './log/checkpoint', type= str,
                        help= "The checkpoint path")
    
    parser.add_argument('--topk', default= '54000', type= int,
                        help= "the evalutation topK")
    
    parser.add_argument('--exp-name', default= 'cifar10-asy', type= str,
                        help= "The experiment name for this run")
    
    parser.add_argument('--test-interval', default=2, type= int ,
                         help= "The interval for perform test")
    
    args = parser.parse_args()
    return args