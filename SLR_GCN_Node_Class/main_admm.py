import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import admm
import numpy as np
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import utils 

# Parse arguments
args = utils.parse_args_admm()

# Set up random seed for Reproducibility
torch.manual_seed(args.seed)


# Set up logger file:
logger = utils.get_logger(args.logging)

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='Cora')
# parser.add_argument('--hidden_channels', type=int, default=16)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
# parser.add_argument('--wandb', action='store_true', help='Track experiment')
# args = parser.parse_args()

# GPU device configuration
device = torch.device("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpus[0])
logger.info('Using device ' + ("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu") +
                     ' for training')


# track the training process
init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
           hidden_channels=args.hidden_channels, device=device)



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
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




def train(ADMM, model, device, data, optimizer, epoch, criterion):
    model.train()
    optimizer.zero_grad()
    ce_loss = None
    mixed_loss = None
    out = model(data.x, data.edge_index, data.edge_weight)
    ce_loss = criterion(out[data.train_mask], data.y[data.train_mask])
    admm.z_u_update(ADMM, model, epoch, args, logger)
    ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(ADMM, model, ce_loss, args.admm_train)
    mixed_loss.backward(retain_graph=True)
    optimizer.step()
    logger.info("cross_entropy loss: {:.4f}, mixed_loss : {:.4f}".format(ce_loss, mixed_loss))
    lossadmm = []
    for k, v in admm_loss.items():
            logger.info("at layer {}, admm loss is {:.6f}".format(k, v))
            lossadmm.append(float(v))

    if args.verbose:
        for k, v in admm_loss.items():
            logger.info("at layer {}, admm loss is {:.6f}".format(k, v))
            ADMM.admmloss[k].extend([float(v)])
    ADMM.ce_prev = ADMM.ce
    ADMM.ce = ce_loss
    return float(mixed_loss), float(ce_loss)


# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index, data.edge_weight)
#     loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    train_acc, val_acc, test_acc = accs
    logger.info('Start training accuracy: {:.3f}, validation accuracy: {:.3f}, test accuracy: {:.3f}.'.format(\
                train_acc, val_acc, test_acc))
    return accs

def main():

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

    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes)

    # load pre-trained model
    model.load_state_dict(torch.load(args.load_model_path, map_location=device))
    model = model.to(device)

    # Set model to GPU device
    model, data = model.to(device), data.to(device)



    criterion = nn.CrossEntropyLoss()
    initial_rho = args.rho

    if args.admm_train:
        # optimizer setup
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=args.lr)  # Only perform weight-decay on first convolution.
        for i in range(args.rho_num):
            logger.info('****************Start ADMM reweighted training****************')
            current_rho = initial_rho * 10 ** i
            if i == 0:
                logger.info(model)
                acc = test(model, data)
            else:
                model.load_state_dict(torch.load(args.model_path + "/model_pruned/pruned_{}_{}_{}_fc1_{}_fc2_{}.pt".format(current_rho/10, \
                    args.config_file, args.sparsity_type, args.fc1weight, args.fc2weight)))
                

            ADMM = admm.ADMM(model, args.config_file_path, args, rho=current_rho)
            admm.admm_initialization(ADMM, model, args.admm_train, args.sparsity_type)  # intialize Z and U variables

            best_val_acc = 0.

            accuracy = []

            if args.optimization == 'admm':
                logger.info("ADMM Training started...")
            if args.optimization == 'savlr':
                logger.info("SLR Training started...")

            for epoch in range(1, args.epochs + 1):
                admm.admm_adjust_learning_rate(optimizer, epoch, args)

                mixed_loss, loss = train(ADMM, model, device, data, optimizer, epoch, criterion)

                train_acc, val_acc, tmp_test_acc = test(model, data)
                accuracy.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc

                logger.info("Best val acc: {:3f}. Test acc: {:3f}.".format(best_val_acc, test_acc))
                if args.check_acc:
                    admm.hard_prune_accuracy_test(ADMM, model, args.sparsity_type, logger, data)

            logger.info("Saving model...")
            torch.save(model.state_dict(), args.model_pruned_path + "/pruned_{}_{}_{}_Conv1Sp_{}_Conv2Sp_{}.pt".format(current_rho/10, \
                    args.config_file, args.sparsity_type, args.conv1linweight, args.conv2linweight))

            logger.info('****************End ADMM reweighted training****************')

    if args.masked_retrain:
        logger.info('****************Start ADMM re-training****************')
        logger.info("Before Hardpruning: ")
        pred = test(model, data)

        #optimizer = optim.SGD(model.parameters(), lr=mylr, momentum=momentum)
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        # optimizer setup
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=args.lr)  # Only perform weight-decay on first convolution.

        ADMM = admm.ADMM(model, file_name=args.config_file_path, args = args, rho=initial_rho)
        admm.hard_prune(ADMM, model, args.sparsity_type, logger)
        compression = admm.test_sparsity(ADMM, model, args.sparsity_type, logger)
        logger.info('Compression rate: ' + str(compression))

        logger.info("After Hardpruning:")
        pred = test(model,data)

        best_val_acc = 0
        best_test_acc = 0
        epoch_loss_dict = []
        testAcc = []
        for epoch in range(1, args.retrain_epoch+1):
            epoch_loss = []
            if args.combine_progressive:
                idx_loss_dict = admm.combined_masked_retrain(ADMM, model, device, data, optimizer, epoch, criterion, args, logger)
            else:
                idx_loss_dict = admm.masked_retrain(ADMM, model, device, data, optimizer, epoch, criterion, args, logger)
            train_acc, val_acc, test_acc = test(model, data)
            if val_acc > (best_val_acc + 0.0015):
                old_best_val_acc = best_val_acc
                old_best_test_acc = best_test_acc
                best_val_acc = val_acc
                best_test_acc = test_acc
                logger.info("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(best_test_acc))
                torch.save(model.state_dict(), args.model_retrained_path+"/Conv1Sp_{}_Conv2Sp_{}_retrained_acc_{:.3f}_testacc_{:.3f}_{}rhos_{}_{}.pt".format(
                               args.conv1linweight, args.conv2linweight, best_val_acc, best_test_acc, args.rho_num, args.config_file, args.sparsity_type))
                logger.info("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(best_test_acc))
                if old_best_val_acc > 0:
                    os.remove(args.model_retrained_path+"/Conv1Sp_{}_Conv2Sp_{}_retrained_acc_{:.3f}_testacc_{:.3f}_{}rhos_{}_{}.pt".format(
                        args.conv1linweight, args.conv2linweight, old_best_val_acc, old_best_test_acc, args.rho_num, args.config_file, args.sparsity_type))

            # for k, v in idx_loss_dict.items():
            #     epoch_loss.append(float(v))
            # epoch_loss = np.sum(epoch_loss)/len(epoch_loss)

            # epoch_loss_dict.append(epoch_loss)
            testAcc.append(test_acc)

            # best_prec1.append(prec1)

        logger.info("After Retraining: ")
        train_acc, val_acc, test_acc = test(model, data)
        admm.test_sparsity(ADMM, model, args.sparsity_type, logger)

        logger.info("Best Accuracy: {:.4f}".format(best_test_acc))
        logger.info('****************End ADMM re-training****************')
     # Save the model checkpoint
    torch.save(model.state_dict(), args.model_final_path+'/model_pruned_Conv1Sp_{}_Conv2Sp_{}_ACC_{}.ckpt'.format(args.sparsity_type, args.conv1linweight, \
                             test_acc))
        
if __name__ == '__main__':
    main()