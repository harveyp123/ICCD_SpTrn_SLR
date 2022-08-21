import argparse
import torch
import logging
import os
def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

def parse_args_dense():
    parser = argparse.ArgumentParser(description='Dense Training for graph neural networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Add Arguments
    parser.add_argument('--input-size', type=int, default=16,
                        help="Input size")
    parser.add_argument('--hidden-size', type=int, default=512,
                        help="Hidden dimension size")
    parser.add_argument('--num-classes', type=int, default=1,
                        help="Hidden dimension size")
    parser.add_argument('--num-epochs', type=int, default=3,
                        help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=1024,
                        help="Batch size")
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help="Learning rate")
    parser.add_argument('--workers', type=int, default=0,
                        help="Number of workers")
    parser.add_argument('--gpus', default='2', help='gpu device ids separated by comma. '
                        '`all` indicates use all gpus.')
    parser.add_argument('--running-dir', default='/home/hop20001/ADMM_SLR_GNN', help='Running directory')
    parser.add_argument('--dataset', default='wiki_talk_dense', help='Dataset name')
    parser.add_argument('--model-name', default='wiki_talk_dense', help='Stored model name')
    parser.add_argument('--log-name', default='wiki_talk_dense_train', help='Log file name')
    parser.add_argument('--lr-decay', type=float, default=0.98,
                        help="Learning rate decay")

    args = parser.parse_args()
    # Dataset directory
    args.dataset_dir = args.running_dir + '/dataset/' + args.dataset + '/'
    # Stored model path
    args.model_path = args.running_dir + '/models/' + args.dataset
    # create path if does not exist
    if not os.path.exists(args.model_path):
          os.makedirs(args.model_path)
          print("The new model directory is created!")
    args.save_model_path = args.model_path + '/' + args.model_name + '.ckpt'
    # Logging path
    args.logging_path = args.running_dir + '/log/' + args.dataset
    if not os.path.exists(args.logging_path):
          os.makedirs(args.logging_path)
          print("The new logging directory is created!")
    args.logging = args.logging_path + '/' + args.log_name + '.log'
    # Parse argument for GPUs
    args.gpus = parse_gpus(args.gpus)
    return args
def parse_args_admm():
    parser = argparse.ArgumentParser(description='ADMM and SLR for graph neural networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Add Arguments
    parser.add_argument('--batch-size', type=int, default=1024,
                        help="batch size ")
    parser.add_argument('--epochs', type=int, default=1,
                        help="Reweighted training epochs")
    parser.add_argument('--retrain-epoch', type=int, default=1,
                        help="Retraining epochs")           
    parser.add_argument('--fc1weight', type=float, default=8/128,
                        help="Layer 1 sparsity")
    parser.add_argument('--fc2weight', type=float, default=8/128,
                        help="Layer 2 sparsity")
    parser.add_argument('--initial_s', type=float, default=0.001,
                        help="Step size for SLR")
    parser.add_argument('--rho', type=float, default=0.01,
                        help="Step size for update multiplier of ADMM/SLR")
    parser.add_argument('--running-dir', default='/home/hop20001/ADMM_SLR_GNN', help='Running directory')
    parser.add_argument('--dataset', default='wiki_talk_dense', help='Dataset name')
    parser.add_argument('--gpus', default='3', help='gpu device ids separated by comma. '
                        '`all` indicates use all gpus.')
    parser.add_argument('--load-model', default='wiki_talk_dense_HID128.ckpt', help='load model name')         
    parser.add_argument('--log-name', default='wiki_talk_dense_admm', help='Log file name')
    parser.add_argument('--optimization', default='savlr', help='Optimization method name')
    parser.add_argument('--check-hardprune-acc', dest='check_acc', action='store_true',
                  help='Evaluate model at reweighted training')
    args = parser.parse_args()
    args.dataset_dir = args.running_dir + '/dataset/' + args.dataset + '/'
    args.model_path = args.running_dir + '/models/' + args.dataset + '/'
    args.load_model_path = args.model_path + args.load_model
    args.logging = args.running_dir + '/log/' + args.dataset + '/' + args.log_name + '_fc1_' \
                                    + str(args.fc1weight) + '_fc2_' + str(args.fc2weight) + '.log'
#     if args.optimization == 'savlr':
#       args.model_pruned_path = args.model_path + '/model_pruned'
#       if not os.path.exists(args.model_pruned_path):
#             os.makedirs(args.model_pruned_path)
#             print("The new directory {} is created!".format(args.model_pruned_path))
#       args.model_retrained_path = args.model_path + '/model_retrained'
#       if not os.path.exists(args.model_retrained_path):
#             os.makedirs(args.model_retrained_path)
#             print("The new directory {} is created!".format(args.model_retrained_path))
#       args.model_final_path = args.model_path + '/model_final'
#       if not os.path.exists(args.model_final_path):
#             os.makedirs(args.model_final_path)
#             print("The new directory {} is created!".format(args.model_final_path))
#     elif args.optimization == 'admm':
#       args.model_pruned_path = args.model_path + '/' + args.optimization + '_model_pruned'
#       if not os.path.exists(args.model_pruned_path):
#             os.makedirs(args.model_pruned_path)
#             print("The new directory {} is created!".format(args.model_pruned_path))
#       args.model_retrained_path = args.model_path + '/' + args.optimization + '_model_retrained'
#       if not os.path.exists(args.model_retrained_path):
#             os.makedirs(args.model_retrained_path)
#             print("The new directory {} is created!".format(args.model_retrained_path))
#       args.model_final_path = args.model_path + '/' + args.optimization + '_model_final'
#       if not os.path.exists(args.model_final_path):
#             os.makedirs(args.model_final_path)
#             print("The new directory {} is created!".format(args.model_final_path))
#     else:
#       print("Optimization name error")
#       exit()
    args.model_pruned_path = args.model_path + '/' + args.optimization + '_model_pruned'
    if not os.path.exists(args.model_pruned_path):
            os.makedirs(args.model_pruned_path)
            print("The new directory {} is created!".format(args.model_pruned_path))
    args.model_retrained_path = args.model_path + '/' + args.optimization + '_model_retrained'
    if not os.path.exists(args.model_retrained_path):
            os.makedirs(args.model_retrained_path)
            print("The new directory {} is created!".format(args.model_retrained_path))
    args.model_final_path = args.model_path + '/' + args.optimization + '_model_final'
    if not os.path.exists(args.model_final_path):
            os.makedirs(args.model_final_path)
            print("The new directory {} is created!".format(args.model_final_path))
    args.seed = 1
    # batch = 1024
    args.test_batch = 1024
    # args.rho = 0.1  # SLR and ADMM both 

    args.rho_num = 1 #define how many rhos for ADMM training
    args.config_file = 'config' #prune config file
    args.sparsity_type = 'irregular' #irregular,column,filter,pattern,random-pattern
    args.momentum = 0.5 #SGD momentum (default: 0.5)
    args.lr_scheduler = 'cosine'


    args.combine_progressive = False #for filter pruning after column pruning
    args.input_size = 16
    args.hidden_size = 128
    args.num_classes = 1
    # args.epochs = 50
    # args.epochs = 1
    args.learning_rate = 0.05
    args.workers = 4
    # args.retrain_epoch = 50# #for retraining
    # args.retrain_epoch = 1# #for retraining

    args.verbose = False #whether to report admm convergence condition
    args.lr_decay = 30 #how many every epoch before lr drop (default: 30)
    args.optmzr = 'sgd' #optimizer used (default: sgd)
    args.log_interval = 9000 #how many batches to wait before logging training status
    args.save_model = "pretrained_mnist.pt" #For Saving the current Model
    args.load_model = None #For loading the model

#     args.optimization = 'savlr' #'admm' or 'savlr'

    args.masked_retrain = True #for masked training
    args.admm_train =  True #for admm training    

    args.admm_epoch = 1 #how often we do admm update
        
    #SAVLR parameters:
    args.M = 200 
    args.r = 0.1
    # args.initial_s = 0.0001
    #SAVLR parameters
    args.config_file_path = args.running_dir + "/profile/" + args.config_file + ".yaml"
    args.gpus = parse_gpus(args.gpus)
    return args


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('GNN')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger