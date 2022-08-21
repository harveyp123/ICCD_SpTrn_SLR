import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import admm
import numpy as np
import os
import utils 
import copy
# Parse arguments
args = utils.parse_args_admm()

# Set up logger file:
logger = utils.get_logger(args.logging)

# Device configuration
device = torch.device("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpus[0])

# home_dir = '/home/hop20001/ADMM-SLR/'
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)


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


def train(ADMM, model, device, train_loader, optimizer, epoch, criterion):
   
    model.train()

    ce_loss = None
    mixed_loss = None
    ctr=0

    total_ce = 0
    log_interval = 1000

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        ctr += 1
        mixed_loss_sum = []
        loss = []
    
        optimizer.zero_grad()
        output = model(data)
     
        ce_loss = criterion(output, target)
        total_ce = total_ce + float(ce_loss.item())
        
        admm.z_u_update(ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, args, logger)  # update Z and U variables

        ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(ADMM, model, ce_loss, args.admm_train)  # append admm losss


        #mixed_loss.backward()
        mixed_loss.backward(retain_graph=True)
        optimizer.step()
 
        mixed_loss_sum.append(float(mixed_loss))
        loss.append(float(ce_loss))


        if batch_idx % log_interval == 0:
            logger.info("cross_entropy loss: {:.6f}, mixed_loss : {:.6f}".format(ce_loss, mixed_loss))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item()))

  
    lossadmm = []
    for k, v in admm_loss.items():
            logger.info("at layer {}, admm loss is {:.6f}".format(k, v))
            lossadmm.append(float(v))

    if args.verbose:
        for k, v in admm_loss.items():
            logger.info("at layer {}, admm loss is {:.6f}".format(k, v))
            ADMM.admmloss[k].extend([float(v)])


    ADMM.ce_prev = ADMM.ce
    ADMM.ce = total_ce / ctr
    
    return mixed_loss_sum, loss


def test(model, device, test_loader):
   with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in test_loader:
        features = features.cuda()
        labels = labels.cuda()

        outputs = model(features)
       
        predicted = torch.round(outputs.data)
     
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    logger.info('Test Accuracy: {:.2f} %'.format(100 * correct / total))
    return (100 * correct / total)



def total_params(model):
        return sum([np.prod(param.size()) for param in model.parameters()])

def param_to_array(param):
    return param.data.numpy().reshape(-1)

def get_sorted_list_of_params(model):
    params = list(model.parameters())
    param_arrays = [param_to_array(param) for param in params]
    return np.sort(np.concatenate(param_arrays))

def main():
     ################################# READ TRAIN DATA #################################
     train_f = torch.jit.load(args.dataset_dir+'train.pt')
     train_l = torch.jit.load(args.dataset_dir+'train_label.pt')

     for key, value in train_f.state_dict().items():
          train_features = value
     for key, value in train_l.state_dict().items():
          train_labels = value

     training_set = Dataset(train_features, train_labels)

     ################################# READ VAL DATA #################################
     val_f = torch.jit.load(args.dataset_dir+'val.pt')
     val_l = torch.jit.load(args.dataset_dir+'val_label.pt')

     for key, value in val_f.state_dict().items():
          val_features = value
     for key, value in val_l.state_dict().items():
          val_labels = value

     val_set = Dataset(val_features, val_labels)
     ################################# READ VAL DATA #################################

     test_f = torch.jit.load(args.dataset_dir+'test.pt')
     test_l = torch.jit.load(args.dataset_dir+'test_label.pt')

     for key, value in test_f.state_dict().items():
          test_features = value
     for key, value in test_l.state_dict().items():
          test_labels = value


     testing_set = Dataset(test_features, test_labels)

     train_loader = torch.utils.data.DataLoader(dataset=training_set, 
                                             batch_size=args.batch_size, 
                                             shuffle=True,
                                             num_workers = args.workers)

     test_loader = torch.utils.data.DataLoader(dataset=testing_set, 
                                             batch_size=args.batch_size, 
                                             shuffle=True,
                                             num_workers = args.workers)



     model = NeuralNet(args.input_size, args.hidden_size, args.num_classes)
     model.load_state_dict(torch.load(args.load_model_path, map_location=device))
     model = model.to(device)
    
     
    #  for i, (name, W) in enumerate(model.named_parameters()):
    #     logger.info(i, "th weight:", name, ", shape = ", W.shape, ", weight.dtype = ", W.dtype)  



     lr = args.learning_rate
     optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
     criterion = nn.BCELoss() #binary cross entropy


     initial_rho = args.rho
     if args.admm_train:
        
        for i in range(args.rho_num):
            logger.info('****************Start ADMM reweighted training****************')
            current_rho = initial_rho * 10 ** i
            if i == 0:
                logger.info(model)
                acc = test(model, device, test_loader)

            else:
                model.load_state_dict(torch.load(args.model_path + "/model_pruned/pruned_{}_{}_{}_fc1_{}_fc2_{}.pt".format(current_rho/10, \
                    args.config_file, args.sparsity_type, args.fc1weight, args.fc2weight)))
                

            ADMM = admm.ADMM(model, args.config_file_path, args, rho=current_rho)
            admm.admm_initialization(ADMM, model, args.admm_train, args.sparsity_type)  # intialize Z and U variables

            best_prec1 = 0.

            accuracy = []

            if args.optimization == 'admm':
                logger.info("ADMM Training started...")
            if args.optimization == 'savlr':
                logger.info("SLR Training started...")

            for epoch in range(1, args.epochs + 1):
                admm.admm_adjust_learning_rate(optimizer, epoch, args)

                mixed_loss, loss = train(ADMM, model, device, train_loader, optimizer, epoch, criterion)

                prec1 = test(model, device, test_loader)
                accuracy.append(prec1)

                best_prec1 = max(prec1, best_prec1)
                if args.check_acc:
                    admm.hard_prune_accuracy_test(ADMM, model, args.sparsity_type, logger, device, test_loader)

            logger.info("Saving model...")
            torch.save(model.state_dict(), args.model_pruned_path + "/pruned_{}_{}_{}_fc1_{}_fc2_{}.pt".format(current_rho/10, \
                    args.config_file, args.sparsity_type, args.fc1weight, args.fc2weight))

            logger.info('****************End ADMM reweighted training****************')
     
     if args.masked_retrain:
        logger.info('****************Start ADMM re-training****************')
        logger.info("Before Hardpruning: ")
        pred = test(model, device, test_loader)

        #optimizer = optim.SGD(model.parameters(), lr=mylr, momentum=momentum)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        ADMM = admm.ADMM(model, file_name=args.config_file_path, args = args, rho=initial_rho)
        admm.hard_prune(ADMM, model, args.sparsity_type, logger)
        compression = admm.test_sparsity(ADMM, model, args.sparsity_type, logger)
        logger.info('Compression rate: ' + str(compression))

        logger.info("After Hardpruning:")
        pred = test(model, device, test_loader)

        best_prec1 = 0
        epoch_loss_dict = []
        testAcc = []
        for epoch in range(1, args.retrain_epoch+1):
            epoch_loss = []
            if args.combine_progressive:
                idx_loss_dict = admm.combined_masked_retrain(ADMM, model, device, train_loader, optimizer, epoch, args, logger)
            else:
                idx_loss_dict = admm.masked_retrain(ADMM, model, device, train_loader, optimizer, epoch, criterion, args, logger)
            prec1 = test(model, device, test_loader)
            if prec1 > (best_prec1 + 0.0015):
                old_best = best_prec1
                best_prec1 = prec1
                logger.info("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(best_prec1))
                torch.save(model.state_dict(),args.model_retrained_path+"/retrained_acc_{:.3f}_{}rhos_{}_{}_fc1_{}_fc2_{}.pt".format(
                               best_prec1, args.rho_num, args.config_file, args.sparsity_type, args.fc1weight, args.fc2weight))
                logger.info("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(old_best))
                if old_best > 0:
                    os.remove(args.model_retrained_path+"/retrained_acc_{:.3f}_{}rhos_{}_{}_fc1_{}_fc2_{}.pt".format(
                        old_best, args.rho_num, args.config_file, args.sparsity_type, args.fc1weight, args.fc2weight))

            for k, v in idx_loss_dict.items():
                epoch_loss.append(float(v))
            epoch_loss = np.sum(epoch_loss)/len(epoch_loss)

            epoch_loss_dict.append(epoch_loss)
            testAcc.append(prec1)

            # best_prec1.append(prec1)

        logger.info("After Retraining: ")
        test(model, device, test_loader)
        admm.test_sparsity(ADMM, model, args.sparsity_type, logger)

        logger.info("Best Accuracy: {:.4f}".format(best_prec1))
        logger.info('****************End ADMM re-training****************')
     # Save the model checkpoint
     torch.save(model.state_dict(), args.model_final_path+'/model_pruned_fc1_{}_fc2_{}_ACC_{}.ckpt'.format(args.fc1weight, \
                            args.fc2weight, prec1))




if __name__ == '__main__':
    main()