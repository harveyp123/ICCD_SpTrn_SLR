seed = 1
batch = 1024
test_batch = 1024
rho = 0.1

rho_num = 1 #define how many rhos for ADMM training
config_file = 'config' #prune config file
sparsity_type = 'irregular' #irregular,column,filter,pattern,random-pattern
momentum = 0.5 #SGD momentum (default: 0.5)
lr_scheduler = 'cosine'


combine_progressive = False #for filter pruning after column pruning
input_size = 16
hidden_size = 128
num_classes = 1
epochs = 50
learning_rate = 0.05
workers = 4
retrain_epoch = 50# #for retraining

verbose = False #whether to report admm convergence condition
lr_decay = 30 #how many every epoch before lr drop (default: 30)
optmzr = 'sgd' #optimizer used (default: sgd)
log_interval = 9000 #how many batches to wait before logging training status
save_model = "pretrained_mnist.pt" #For Saving the current Model
load_model = None #For loading the model

optimization = 'savlr' #'admm' or 'savlr'

masked_retrain = True #for masked training
admm_train =  True #for admm training    

admm_epoch = 1 #how often we do admm update
    
#SAVLR parameters:
M = 200
r = 0.1
initial_s = 0.0001
     
