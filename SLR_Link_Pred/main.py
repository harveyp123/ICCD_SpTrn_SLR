import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.autograd.profiler as profiler
from torch.autograd import Function
import utils 
from torch.optim.lr_scheduler import LambdaLR

# Parse arguments
args = utils.parse_args_dense()


# Set up logger file:
logger = utils.get_logger(args.logging)

# GPU device configuration
device = torch.device("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu")


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

# home_dir = '/home/hop20001/ADMM-SLR/'

################################# READ TRAIN DATA #################################
train_f = torch.jit.load(args.dataset_dir + 'train.pt')
train_l = torch.jit.load(args.dataset_dir + 'train_label.pt')

for key, value in train_f.state_dict().items():
     train_features = value
for key, value in train_l.state_dict().items():
     train_labels = value

# train_features = train_features[0:100000]
# train_labels = train_labels[0:100000]

# print(train_labels.shape)
# print(train_features.shape)
# print(train_labels)

training_set = Dataset(train_features, train_labels)

################################# READ VAL DATA #################################
val_f = torch.jit.load(args.dataset_dir + 'val.pt')
val_l = torch.jit.load(args.dataset_dir + 'val_label.pt')

for key, value in val_f.state_dict().items():
     val_features = value
for key, value in val_l.state_dict().items():
     val_labels = value



# print(val_labels.shape)
# print(val_features.shape)
# print(val_labels)
# print(val_features)

val_set = Dataset(val_features, val_labels)
################################# READ VAL DATA #################################

test_f = torch.jit.load(args.dataset_dir + 'test.pt')
test_l = torch.jit.load(args.dataset_dir + 'test_label.pt')

for key, value in test_f.state_dict().items():
     test_features = value
for key, value in test_l.state_dict().items():
     test_labels = value



# print(test_labels.shape)
# print(test_features.shape)
# print(test_labels)
# print(test_features)

testing_set = Dataset(test_features, test_labels)




# Data loader
train_loader = torch.utils.data.DataLoader(dataset=training_set, 
                                           batch_size=args.batch_size, 
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers = args.workers)

test_loader = torch.utils.data.DataLoader(dataset=testing_set, 
                                          batch_size=args.batch_size, 
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers = args.workers)



model = NeuralNet(args.input_size, args.hidden_size, args.num_classes).to(device)

# Loss and optimizer

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
criterion = nn.BCELoss() #binary cross entropy
total_step = len(train_loader)

logger.info('****************Start training****************')

## Change learning rate of optimizer:

# Naive way to change the learning rate
# for g in optim.param_groups:
#     g['lr'] = 0.001

#Change learning rate using torch package
lambda_para = lambda epoch: args.lr_decay ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda_para])

Best_ACC = 0
for epoch in range(args.num_epochs):
     # Training phase
     for i, (features, labels) in enumerate(train_loader):  

          # features = features.cuda()
          # labels = labels.cuda()
          features = features.to(device)
          labels = labels.to(device)
          outputs = model(features)
          loss = criterion(outputs, labels)
         
          optimizer.zero_grad()
          loss.backward()
 
          optimizer.step()
          
          
          if (i+1) % 1000 == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, args.num_epochs, i+1, total_step, loss))
     # Adjust the learning rate
     scheduler.step()

     # In test phase, we don't need to compute gradients (for memory efficiency)
     with torch.no_grad():
          correct = 0
          total = 0
          for features, labels in test_loader:
               features = features.to(device)
               labels = labels.to(device)
               outputs = model(features)
               predicted = torch.round(outputs.data)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
          ACC = correct / total
          logger.info('Current {}th epoch  Accuracy: {} %'.format(epoch, 100 * ACC))
          if ACC > Best_ACC:
               Best_ACC = ACC
               # Save the model checkpoint for best model
               torch.save(model.state_dict(), args.save_model_path)
               logger.info('Saved model to ' + args.save_model_path)
          logger.info('Best  Accuracy: {} %'.format(100 * Best_ACC))
          
logger.info('****************End training****************')

