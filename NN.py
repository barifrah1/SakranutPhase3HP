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


class Net(nn.Module):

    def __init__(self, feature_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_num, 256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


def predict(x, y, model, auc_list, loss_list):
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        x_var = Variable(torch.FloatTensor(x))
        output = model(x_var)
        #test_results = probs(output)
        loss = criterion(output.flatten(), torch.tensor(y).float())
        loss_list.append(loss)
        #values, labels = torch.max(test_result, 1)
        auc = roc_auc_score(y, output.numpy())
        auc_list.append(auc)
    return auc, auc_list, loss, loss_list


def train(X_train, y_train, model, x_test, y_test,
          batch_size,
          n_epochs,
          criterion=nn.BCELoss(),
          ):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch_no = len(X_train) // batch_size
    # print(batch_no)
    train_loss = 0
    auc_test = []
    auc_train = []
    test_loss_list = []
    #probs = nn.Softmax(dim=1)
    #train_loss_min = np.Inf
    for epoch in tqdm(range(n_epochs)):
        for i in range(batch_no):
            model.train()
            start = i*batch_size
            end = start+batch_size
            x_var = Variable(torch.FloatTensor(X_train[start:end]))
            y_var = Variable(torch.LongTensor(y_train[start:end]))
            optimizer.zero_grad()
            output = model(x_var)
            #softmax_output = probs(output)
            loss = criterion(output.flatten(), y_var.float())
            loss.backward()
            optimizer.step()
            #values, labels = torch.max(output, 1)
            #num_right = np.sum(labels.data.numpy() == y_train[start:end])
            train_loss += loss.item()*batch_size

        train_loss = train_loss / len(X_train)
        # print('labels',labels)
        # print('y_train',y_train[start:end])
        with torch.no_grad():
            model.eval()
            x_var = Variable(torch.FloatTensor(X_train))
            output_epoch = model(x_var)
            #softmax_output = probs(output_epoch)
            #values, labels = torch.max(output_epoch, 1)
            auc = roc_auc_score(y_train, output_epoch.numpy())
            auc_train.append(auc)
        #print('auc', auc)
        """if train_loss <= train_loss_min:
            print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))
            torch.save(model.state_dict(), "model.pt")
            train_loss_min = train_loss
        """

        """if epoch:
            print('')
            print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {} \tTrain auc:".format(
                epoch+1, train_loss, auc))  # , num_right / len(y_train[start:end])"""
        if epoch % 10 == 0:
            print("train loss is: ", train_loss)
            pre, auc_test, test_loss, test_loss_list = predict(
                x_test, y_test, model, auc_test, test_loss_list)
            print(f'predict auc in epoch {epoch+1} is {pre}')
            print(f'predict test_loss in epoch {epoch+1} is {test_loss}')
    print(" Train auc: {} \tTest AUC: {}".format(
        auc_train, auc_test))  # , num_right / len(y_train[start:end])
    print(" test_loss: {} ".format(
        test_loss_list))
    print('Training Ended! ')
    plt.plot(test_loss_list)
    plt.show()
    plt.plot(auc_test)
    plt.show()
