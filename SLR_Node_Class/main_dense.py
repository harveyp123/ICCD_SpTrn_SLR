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
torch.cuda.set_device(args.gpus[0])

# Fully connected neural network with two hidden layers
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

# home_dir = '/home/hop20001/ADMM-SLR/'

################################# READ TRAIN DATA #################################
train_f = torch.jit.load(args.dataset_dir + 'train.pt')
train_l = torch.jit.load(args.dataset_dir + 'train_label.pt')

for key, value in train_f.state_dict().items():
     train_features = value
for key, value in train_l.state_dict().items():
     train_labels = value

train_labels = torch.flatten(train_labels)

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

val_labels = torch.flatten(val_labels)

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

test_labels = torch.flatten(test_labels)

# print(test_labels.shape)
# print(test_features.shape)
# print(test_labels)
# print(test_features)

testing_set = Dataset(test_features, test_labels)




# Data loader
train_loader = torch.utils.data.DataLoader(dataset=training_set, 
                                           batch_size=args.batch_size, 
                                        #    shuffle=True,
                                        #    pin_memory=True,
                                           num_workers = args.workers)

test_loader = torch.utils.data.DataLoader(dataset=testing_set, 
                                          batch_size=args.batch_size, 
                                        #   shuffle=True,
                                        #   pin_memory=True,
                                          num_workers = args.workers)



model = NeuralNet(args.input_size, args.hidden1_size, args.hidden2_size, args.num_classes).to(device)

# Loss and optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
total_step = len(train_loader)

logger.info('****************Start training****************')

## Change learning rate of optimizer:

# Naive way to change the learning rate
# for g in optim.param_groups:
#     g['lr'] = 0.001

#Change learning rate using torch package
lambda_para = lambda epoch: args.lr_decay ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda_para])

model.train()
Best_ACC = 0
for epoch in range(args.num_epochs):
     # Training phase
     for i, (features, labels) in enumerate(train_loader):  

          # features = features.cuda()
          # labels = labels.cuda()
          features = features.cuda()
          labels = labels.cuda()
          outputs = model(features)
          # print(labels.shape, '\n', outputs.shape)
          loss = criterion(outputs, labels)
         
          optimizer.zero_grad()
          loss.backward()
 
          optimizer.step()
          
          
          if (i+1) % 6 == 0:
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
               predicted = outputs.argmax(1)
               # logger.info(predicted)
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

