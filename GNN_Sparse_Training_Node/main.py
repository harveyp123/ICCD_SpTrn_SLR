from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy
import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay, LinearDecayTheta, CosineDecayTheta
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size, bias=False) 
        self.dropout1 = nn.Dropout(0.1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size, bias=False) 
        self.dropout2 = nn.Dropout(0.1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes, bias=False) 
        self.dropout3 = nn.Dropout(0.1)
        self.relu3 = nn.ReLU()
       # self.softmax = nn.LogSoftmax()

    
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.dropout3(out)
        out = self.relu3(out)
        return nn.functional.log_softmax(out, 1)
       # return self.softmax(out)



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels):
        'Initialization'
        self.labels = labels
        self.features = features

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.features[index]
        y = self.labels[index]

        return X, y



def train(args, model, device, train_loader, optimizer, epoch, mask=None):
   
    model.train()

    ce_loss = None
    log_interval = 1000

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        # loss = []
    
        optimizer.zero_grad()
        output = model(data)

        criterion = nn.NLLLoss()
        # print("type(output): ", type(output))
        # print("type(target): ", target.dtype)
        # target = target.float()
        # print("type(target) after casting: ", target.dtype)
        loss = criterion(output, target)
        loss.backward()
        if mask is not None: mask.step()
        else: optimizer.step()

        # loss.append(float(ce_loss))

        if batch_idx % log_interval == 0:
            # print("cross_entropy loss: {:.6f}".format(ce_loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))



def test(args, model, device, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            features = features.cuda()
            labels = labels.cuda()

            outputs = model(features)
            
            # predicted = torch.round(outputs.data)
            predicted = outputs.argmax(1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy: {:.2f} %'.format(100 * correct / total))
        return (100 * correct / total)



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
        total_zeros += zeros
        nonzeros = np.sum(W != 0)
        total_nonzeros += nonzeros
        print("sparsity at layer {} is {}".format(name, float(zeros) / (float(zeros + nonzeros))))


def main():
    # Training settings
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
    parser.add_argument('--data_path', type=str)


    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))


        ################################# READ TRAIN DATA #################################
        train_f = torch.jit.load(args.data_path + 'train.pt')
        train_l = torch.jit.load(args.data_path + 'train_label.pt')

        for key, value in train_f.state_dict().items():
            train_features = value
        for key, value in train_l.state_dict().items():
            train_labels = value

        train_labels = torch.flatten(train_labels)
        print(train_labels.shape)
        print(train_features.shape)

        training_set = Dataset(train_features, train_labels)

        ################################# READ VAL DATA #################################
        val_f = torch.jit.load(args.data_path + 'val.pt')
        val_l = torch.jit.load(args.data_path + 'val_label.pt')

        for key, value in val_f.state_dict().items():
            val_features = value
        for key, value in val_l.state_dict().items():
            val_labels = value

        val_labels = torch.flatten(val_labels)
        print(val_labels.shape)
        print(val_features.shape)

        val_set = Dataset(val_features, val_labels)
        ################################# READ VAL DATA #################################

        test_f = torch.jit.load(args.data_path + 'test.pt')
        test_l = torch.jit.load(args.data_path + 'test_label.pt')

        for key, value in test_f.state_dict().items():
            test_features = value
        for key, value in test_l.state_dict().items():
            test_labels = value

        test_labels = torch.flatten(test_labels)
        print(test_labels.shape)
        print(test_features.shape)


        testing_set = Dataset(test_features, test_labels)

        train_loader = torch.utils.data.DataLoader(dataset=training_set, 
                                                batch_size=args.batch_size, 
                                                #shuffle=True,
                                                # pin_memory=True,
                                                num_workers = 1, sampler = torch.utils.data.RandomSampler(training_set))

                                                

        test_loader = torch.utils.data.DataLoader(dataset=testing_set, 
                                                batch_size=args.batch_size, 
                                                #shuffle=True,
                                                #pin_memory=True,
                                                num_workers = 1, sampler = torch.utils.data.RandomSampler(testing_set))



        # model = NeuralNet(args.input_size, args.hidden_size, args.num_classes).to(device)
        model = NeuralNet(args.input_size, args.hidden1_size, args.hidden2_size, args.num_classes).to(device)
        # model.load_state_dict(torch.load("model.ckpt"))

        print_and_log(model)
        print_and_log('=' * 60)
        # print_and_log(args.model)
        print_and_log('=' * 60)

        print_and_log('=' * 60)
        print_and_log('Prune mode: {0}'.format(args.death))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('=' * 60)


        # optimizer = None
        # if args.optimizer == 'sgd':
        #     optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
        # elif args.optimizer == 'adam':
        #     optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        # else:
        #     print('Unknown optimizer: {0}'.format(args.optimizer))
        #     raise Exception('Unknown optimizer.')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)


        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=4e-08)
        


        # if args.fp16:
        #     print('FP16')
        #     optimizer = FP16_Optimizer(optimizer,
        #                                static_loss_scale = None,
        #                                dynamic_loss_scale = True,
        #                                dynamic_loss_args = {'init_scale': 2 ** 16})
        #     model = model.half()

        mask = None
        if args.sparse:
            decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))
            print("length of train_loader: ", len(train_loader))
            decay_theta = LinearDecayTheta(args.theta, args.factor, args.theta_decay_freq)
            # decay_theta = CosineDecayTheta(args.theta, len(train_loader)*(args.epochs*args.multiplier))
            # print("args.theta_min: ", args.theta_min)
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, theta_decay=decay_theta, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, theta=args.theta, epsilon=args.epsilon, args=args)
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

        best_acc = 0.0

        for epoch in range(1, args.epochs*args.multiplier + 1):
            t0 = time.time()
            train(args, model, device, train_loader, optimizer, epoch, mask)
            # lr_scheduler.step()
            if args.valid_split > 0.0:
                val_acc = test(args, model, device, test_loader)
                
                # test(args, model, device, test_loader, is_test_set=True)

            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                torch.save(model.state_dict(), args.save)

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
        print('Testing model')
        model.load_state_dict(torch.load(args.save))
        test(args, model, device, test_loader)
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))

        layer_fired_weights, total_fired_weights = mask.fired_masks_update()
        for name in layer_fired_weights:
            print('The final percentage of fired weights in the layer', name, 'is:', layer_fired_weights[name])
        print('The final percentage of the total fired weights is:', total_fired_weights)
        test_sparsity(model)

if __name__ == '__main__':
   main()
