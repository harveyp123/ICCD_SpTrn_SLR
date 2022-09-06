from __future__ import print_function

import argparse
import os
import sys
import os.path as osp
import time

import torch
import torch.nn.functional as F
import logging
import hashlib
import copy

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import utils 
import numpy as np

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay, LinearDecayTheta, CosineDecayTheta
import torch.backends.cudnn as cudnn


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None


class GCN(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=not args.use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

# def train(args, model, device, train_loader, optimizer, epoch, mask=None):
   
#     model.train()
#     optimizer.zero_grad()

#     # ce_loss = None
#     # log_interval = 1000

#     for batch_idx, (data, target) in enumerate(train_loader):
#         data = data.cuda()
#         target = target.cuda()
#         # loss = []
    
#         optimizer.zero_grad()
#         output = model(data)

#         criterion = nn.NLLLoss()
#         # print("type(output): ", type(output))
#         # print("type(target): ", target.dtype)
#         # target = target.float()
#         # print("type(target) after casting: ", target.dtype)
#         loss = criterion(output, target)
#         loss.backward()
#         if mask is not None: mask.step()
#         else: optimizer.step()

#         # loss.append(float(ce_loss))

#         if batch_idx % log_interval == 0:
#             # print("cross_entropy loss: {:.6f}".format(ce_loss))
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))



def train(model, optimizer, data, mask=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # optimizer.step()
    if mask is not None: mask.step()
    else: optimizer.step()
    return float(loss)


# @torch.no_grad()

def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def total_params(model):
        return sum([np.prod(param.size()) for param in model.parameters()])

def param_to_array(param):
    return param.data.numpy().reshape(-1)

def get_sorted_list_of_params(model):
    params = list(model.parameters())
    param_arrays = [param_to_array(param) for param in params]
    return np.sort(np.concatenate(param_arrays))

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}.log'.format(args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def test_sparsity(model):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    total_zeros = 0
    total_nonzeros = 0

    for i, (name, W) in enumerate(model.named_parameters()):
        
        if 'bias' in name:
            continue
        W = W.cpu().detach().numpy()
        zeros = np.sum(W == 0)
        print("zeros: ", zeros)
        total_zeros += zeros
        nonzeros = np.sum(W != 0)
        print("nonzeros: ", nonzeros)
        total_nonzeros += nonzeros
        print("sparsity at layer {} is {}".format(name, float(zeros) / (float(zeros + nonzeros))))



def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=1024, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=18, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    # parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--decay_frequency', type=int, default=30000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    # parser.add_argument('--model', type=str, default='vgg-c')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--input_size', type=int, default=64, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--hidden1_size', type=int, default=256, metavar='N',
                        help='number of epochs to train (default: 256)')
    parser.add_argument('--hidden2_size', type=int, default=128, metavar='N',
                        help='number of epochs to train (default: 128)')
    parser.add_argument('--num_classes', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    # parser.add_argument('--batch', type=int, default=512, metavar='N',
    #                     help='number of epochs to train (default: 512)')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name: Cora, Pubmed, or CiteSeer')
    parser.add_argument('--hidden_channels', type=int, default=16)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--wandb', action='store_true', help='Track experiment')

    parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                        '`all` indicates use all gpus.')
    # parser.add_argument('--seed', default='0', help='Set random seed for Reproducibility')              
    parser.add_argument('--running-dir', type=str, default=osp.dirname(osp.realpath(__file__)), help='Running directory')

    parser.add_argument('--model-name', default='Cora_dense', help='Stored model name')
    parser.add_argument('--log-name', default='Cora_dense_train', help='Log file name')
    parser.add_argument('--data_path', type=str)

    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()

    # args = utils.parse_args_dense()
    args.dataset_dir = args.running_dir + '/dataset/' 
    # Stored model path
    args.model_pt = args.running_dir + '/models/'
    args.model_path = args.model_pt + args.dataset

    # create path if does not exist
    if not os.path.exists(args.model_pt):
          os.makedirs(args.model_pt)
          print("The new model directory is created!")

    # create path if does not exist
    if not os.path.exists(args.model_path):
          os.makedirs(args.model_path)
          print("The new dataset model directory is created!")
    args.save_model_path = args.model_path + '/' + args.model_name + '.ckpt'

    # Logging path
    args.logging_pt = args.running_dir + '/log/'
    args.logging_path = args.logging_pt + args.dataset
    if not os.path.exists(args.logging_pt):
          os.makedirs(args.logging_pt)
          print("The new logging directory is created!")
    if not os.path.exists(args.logging_path):
          os.makedirs(args.logging_path)
          print("The new logging dataset directory is created!")
    args.logging = args.logging_path + '/' + args.log_name + '.log'
    # Parse argument for GPUs
    args.gpus = parse_gpus(args.gpus)

    setup_logger(args)
    print_and_log(args)

    # Set up random seed for Reproducibility
    torch.manual_seed(args.seed)

    # Set up logger file:
    logger = utils.get_logger(args.logging)



    # # GPU device configuration
    # device = torch.device("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(args.gpus[0])
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info('Using device ' + ("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu") +
                        ' for training')


    # track the training process
    init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
            hidden_channels=args.hidden_channels, device=device)

    # dataset downloading/configuration
    dataset = Planetoid(args.dataset_dir, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]



    if args.use_gdc:
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        data = transform(data)





    model = GCN(args, dataset.num_features, args.hidden_channels, dataset.num_classes)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.

    print_and_log(model)
    print_and_log('=' * 60)
    # print_and_log(args.model)
    print_and_log('=' * 60)

    print_and_log('=' * 60)
    print_and_log('Prune mode: {0}'.format(args.death))
    print_and_log('Growth mode: {0}'.format(args.growth))
    print_and_log('Redistribution mode: {0}'.format(args.redistribution))
    print_and_log('=' * 60)

    mask = None
    if args.sparse:
        decay = CosineDecay(args.death_rate, args.epochs*args.multiplier)
        # print("length of train_loader: ", len(train_loader))
        decay_theta = LinearDecayTheta(args.theta, args.factor, args.theta_decay_freq)
        # decay_theta = CosineDecayTheta(args.theta, len(train_loader)*(args.epochs*args.multiplier))
        # print("args.theta_min: ", args.theta_min)
        mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, theta_decay=decay_theta, growth_mode=args.growth,
                        redistribution_mode=args.redistribution, theta=args.theta, epsilon=args.epsilon, args=args)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

    best_val_acc = 0.0

    # best_acc = 0.0

    # for epoch in range(1, args.epochs*args.multiplier + 1):
    #     t0 = time.time()
    #     train(args, model, optimizer, data, mask)
    #     # train(args, model, optimizer, data)
    #     # lr_scheduler.step()
    #     if args.valid_split > 0.0:
    #         val_acc = test(args, model, device, test_loader)
            
    #         # test(args, model, device, test_loader, is_test_set=True)

    #     if val_acc > best_acc:
    #         print('Saving model')
    #         best_acc = val_acc
    #         torch.save(model.state_dict(), args.save)

    logger.info('****************Start training****************')
    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, data, mask)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            torch.save(model.state_dict(), args.save_model_path)
            logger.info('Saved model to ' + args.save_model_path + ' . Validation accuracy: {:.3f}, test accuracy: {:.3f}.'.format(\
                    val_acc, test_acc))
        logger.info('In the {}th epoch, the loss is: {:.3f}, training accuracy: {:.3f}, validation accuracy: {:.3f}, test accuracy: {:.3f}.'.format(\
            epoch, loss, train_acc, val_acc, test_acc))
        

    logger.info('****************End training****************')

    logger.info('Testing model')
    model.load_state_dict(torch.load(args.save_model_path))
    test(model, data)
    logger.info('Final test: ' + ' . Validation accuracy: {:.3f}, test accuracy: {:.3f}.'.format(\
                    val_acc, test_acc))
    # print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))

    layer_fired_weights, total_fired_weights = mask.fired_masks_update()
    for name in layer_fired_weights:
        print('The final percentage of fired weights in the layer', name, 'is:', layer_fired_weights[name])
    print('The final percentage of the total fired weights is:', total_fired_weights)
    test_sparsity(model)

if __name__ == '__main__':
   main()