import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from data_loader import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from matplotlib import pyplot as plt
from tqdm import tqdm
from hyperparameters import HyperParameters


def net_build(feature_num):
    num_layer = np.random.randint(2, 15, size=1)[0]
    print('number layer in current network', num_layer)
    layer = []
    last = 0
    for x in range(num_layer):
        hidden_size = np.random.randint(10, 300, size=1)[0]
        print("hidden_size of layer{} is {}.".format(x+1, hidden_size))
        drop_rand = np.random.uniform(0, 1)
        if x == 0:
            layer.append(nn.Linear(feature_num, hidden_size))
            # print("layer{} is {}x{}".format(x+1, feature_num,hidden_size))
            if drop_rand < 0.2:
                drop_size = np.random.uniform(0.2, 0.9)
                layer.append(nn.Dropout(drop_size))
                layer.append(nn.ReLU())
                print("dropout after layer{} with p={}".format(x+1, drop_size))
            last = hidden_size
            continue
        if x == num_layer-1:
            layer.append(nn.Linear(last, 1))
            # print("layer{} is {}x{}".format(x+1, last,1))
            continue
        layer.append(nn.Linear(last, hidden_size))
        # print("layer{} is {}x{}".format(x+1, last,hidden_size))
        layer.append(nn.ReLU())
        if drop_rand < 0.2:
            drop_size = np.random.uniform(0.2, 0.9)
            layer.append(nn.Dropout(drop_size))
            print("dropout after layer{} with p={}".format(x+1, drop_size))
        last = hidden_size
    layer.append(nn.Sigmoid())
    net = nn.Sequential(*layer)
    return net


class Net(nn.Module):

    # in case building the net for the final model
    def __init__(self, feature_num, net=None, dropout=0.5):
        super(Net, self).__init__()
        if(net == None):
            self.seq = nn.Sequential(
                nn.Linear(feature_num, 32),
                nn.BatchNorm1d(32),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(32, 6),
                nn.ReLU(),
                nn.Linear(6, 1),
                nn.Sigmoid())
        else:
            self.seq = net

    def forward(self, x):
        return self.seq(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(x, y, model, auc_list=None, loss_list=None):
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        x_var = Variable(torch.FloatTensor(x))
        output = model(x_var)
        loss = criterion(output.flatten(), torch.tensor(y).float()).item()
        #auc = roc_auc_score(y, output.numpy())
        if loss_list != None:
            loss_list.append(loss)
        """if auc_list != None:
            auc_list.append(auc)"""
    return auc, auc_list, loss, loss_list


def train(X_train, y_train, model, X_val, y_val,
          hyper_params,
          criterion=nn.BCELoss()
          ):
    lr = hyper_params[HyperParameters.hyperparams_dict['lr']]
    batch_size = hyper_params[HyperParameters.hyperparams_dict['batch_size']]
    n_epochs = hyper_params[HyperParameters.hyperparams_dict['n_epochs']]
    weight_decay = hyper_params[HyperParameters.hyperparams_dict['weight_decay']]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    batch_no = len(X_train) // batch_size  # numebr of batches per epoch
    train_loss = 0
    auc_test = 0
    auc_train = []
    auc_validation = []
    validation_loss_list = []
    train_loss_list = []
    # calculate first error on validation before training
    untrained_val_auc, auc_validation, untrained_val_loss, validation_loss_list = predict(
        X_val, y_val, model, auc_validation, validation_loss_list)
    model.eval()
    with torch.no_grad():
        x_var = Variable(torch.FloatTensor(X_train))
        train_loss_list.append(
            criterion(model(x_var).flatten(), torch.tensor(y_train).float()).item())
    threshold = 0
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for i in range(batch_no):
            start = i*batch_size
            end = start+batch_size
            x_var = Variable(torch.FloatTensor(X_train[start:end]))
            y_var = Variable(torch.LongTensor(y_train[start:end]))
            optimizer.zero_grad()
            output = model(x_var)
            loss = criterion(output.flatten(), y_var.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
        train_loss = train_loss / len(X_train)
        train_loss_list.append(train_loss)
        model.eval()
        with torch.no_grad():
            x_var = Variable(torch.FloatTensor(X_val))
            y_var = Variable(torch.FloatTensor(y_val))
            trained_val_auc, auc_validation, trained_val_loss, validation_loss_list = predict(
                X_val, y_val, model, auc_validation, validation_loss_list)

        current = validation_loss_list[-1]
        if len(validation_loss_list) > 1:
            previews = validation_loss_list[len(validation_loss_list)-2]
            if (previews-current) < 1e-3:
                threshold += 1
            if (previews-current) >= 1e-3:
                if threshold > 0:
                    threshold -= 1
        if threshold == 4:
            break

    print('best_loss for current iteration:', validation_loss_list[-1])
    return validation_loss_list[-1], epoch
