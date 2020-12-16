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
        self.fc1 = nn.Linear(feature_num, 150)
        self.dropout = nn.Dropout(0.5)
        #self.bn1 = nn.BatchNorm1d(num_features=15)
        self.fc2 = nn.Linear(150, 60)
        self.fc3 = nn.Linear(60, 30)
        self.fc4 = nn.Linear(30, 10)
        self.fc5 = nn.Linear(10, 1)
        #self.dropout = nn.Dropout(0.5)
        #self.fc3 = nn.Linear(10, 5)
        #self.fc4 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.dropout(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        #x = self.dropout(x)
        x = self.sigmoid(self.fc5(x))
        return x


def predict(x, y, model, auc_list=None, loss_list=None):
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        x_var = Variable(torch.FloatTensor(x))
        output = model(x_var)
        loss = criterion(output.flatten(), torch.tensor(y).float())
        auc = roc_auc_score(y, output.numpy())
        if loss_list != None:
            loss_list.append(loss)
        if auc_list != None:
            auc_list.append(auc)
    return auc, auc_list, loss, loss_list


def train(X_train, y_train, model, X_val, y_val, x_test, y_test,
          batch_size,
          n_epochs,
          criterion=nn.BCELoss(),
          ):
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-9)
    batch_no = len(X_train) // batch_size  # numer of batches per epoch
    train_loss = 0
    auc_test = []
    auc_train = []
    auc_validation = []
    validation_loss_list = []
    # calculate first error on test before training
    untrained_test_auc, _, untrained_test_loss, _ = predict(
        x_test, y_test, model)
    print(
        f"Test loss before training is: {untrained_test_loss}")
    print(
        f"Test auc before training is: {untrained_test_auc}")
    # calculate first error on validation before training
    untrained_val_auc, auc_validation, untrained_val_loss, validation_loss_list = predict(
        X_val, y_val, model, auc_validation, validation_loss_list)
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
        model.eval()
        with torch.no_grad():
            x_var = Variable(torch.FloatTensor(X_val))
            y_var = Variable(torch.FloatTensor(y_val))
            output_epoch = model(x_var)
            trained_val_auc, auc_validation, trained_val_loss, validation_loss_list = predict(
                X_val, y_val, model, auc_validation, validation_loss_list)
        """if train_loss <= train_loss_min:
            print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))
            torch.save(model.state_dict(), "model.pt")
            train_loss_min = train_loss
        """
        if epoch % 5 == 0:
            print(f" Current Train loss on epoch {epoch+1} is: ", train_loss)
            print(
                f"Current validation loss on epoch {epoch+1} is: {trained_val_loss} \n Current validation auc is: { trained_val_auc} ")
    print("Validation loss by epoch: {} ".format(
        validation_loss_list))
    print("Validation AUC by epoch: {} ".format(
        auc_train, auc_validation))
    trained_test_auc, _, trained_test_loss, _ = predict(
        x_test, y_test, model)
    print(" Test_loss: {} and Test auc: {}".format(
        trained_test_loss, trained_test_auc))
    print('Training Ended!! ')
    plt.plot(validation_loss_list)
    plt.show()
    plt.plot(auc_validation)
    plt.show()
