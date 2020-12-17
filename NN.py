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




def net_build(feature_num):
    num_layer=np.random.randint(2,15, size=1)[0]
    print('number layer in current network',num_layer)
    layer=[]
    last=0
    for x in range(num_layer):
        hidden_size=np.random.randint(10,300, size=1)[0]
        print("hidden_size of layer{} is {}.".format(x+1, hidden_size))
        drop_rand=np.random.uniform(0,1)
        if x==0:
            layer.append(nn.Linear(feature_num,hidden_size))
            
            print("layer{} is {}x{}".format(x+1, feature_num,hidden_size))
            if drop_rand<0.2:
                drop_size=np.random.uniform(0.2,0.9)
                layer.append(nn.Dropout(drop_size))
                layer.append(nn.ReLU())
                print("dropout after layer{} with p={}".format(x+1,drop_size))
            last=hidden_size 
            continue
        if x==num_layer-1:
            layer.append(nn.Linear(last,1))
            print("layer{} is {}x{}".format(x+1, last,1))
            continue 
        layer.append(nn.Linear(last,hidden_size))
        print("layer{} is {}x{}".format(x+1, last,hidden_size))
        layer.append(nn.ReLU())
        if drop_rand<0.2:
            drop_size=np.random.uniform(0.2,0.9)
            layer.append(nn.Dropout(drop_size)) 
            print("dropout after layer{} with p{}".format(x+1,drop_size))
        last=hidden_size
    layer.append(nn.Sigmoid())    
    net = nn.Sequential(*layer)
    return net

class Net(nn.Module):

    def __init__(self, feature_num):
        super(Net, self).__init__()
        """layer_size=np.random.randint(10,300, size=1)[0]
        layer_size=layer_size[0]
        drop_size=np.random.uniform(0.2,0.9)
        print('layer_size',layer_size)
        print('drop_size',drop_size)
        self.fc1 = nn.Linear(feature_num, layer_size)
        self.dropout = nn.Dropout(drop_size)
        #self.bn1 = nn.BatchNorm1d(num_features=15)
        self.fc2 = nn.Linear(layer_size, int(layer_size*0.7))
        self.fc3 = nn.Linear(int(layer_size*0.7), int(layer_size*0.5))
        self.fc4 = nn.Linear(int(layer_size*0.5), int(layer_size*0.3))
        self.fc5 = nn.Linear(int(layer_size*0.3), 1)
        #self.dropout = nn.Dropout(0.5)
        #self.fc3 = nn.Linear(10, 5)
        #self.fc4 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(0.5)
        """
        self.seq=net_build(feature_num)
    def forward(self,x):
        return self.seq(x)   


    """def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.dropout(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        #x = self.dropout(x)
        x = self.sigmoid(self.fc5(x))
        return x
    """


def param ():
    batch_size=np.random.randint(256,50000, size=1)
    lr=np.random.uniform(1e-6,0.01)
    print('lr',lr)
    print('batch_size',batch_size)
    return lr,batch_size[0]

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
          batch_size,lr,
          n_epochs,
          criterion=nn.BCELoss()
          ):
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    weight_decay=np.random.uniform(lr/10000,lr/10)
    print('weight_decay',weight_decay)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
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
    treshold=0
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
        current=validation_loss_list[-1]
        if len(validation_loss_list)>1:
            previews=validation_loss_list[len(validation_loss_list)-2]
            if (previews-current)<1e-5:
                treshold+=1
            if (previews-current)>=1e-5:
                treshold=0
        if treshold==4:
            ('break in epoch', epoch+1 )
            break
                
            
    print("Validation loss by epoch: {} ".format(
        validation_loss_list))
    print('best_loss for current iteration:',min(validation_loss_list))
    #print("Validation AUC by epoch: {} ".format(
    #    auc_train, auc_validation))
    #trained_test_auc, _, trained_test_loss, _ = predict(
    #    x_test, y_test, model)
    #print(" Test_loss: {} and Test auc: {}".format(
    #    trained_test_loss, trained_test_auc))
    #print('Training Ended!! ')
    #plt.plot(validation_loss_list)
    #plt.show()
    #plt.plot(auc_validation)
    #plt.show()
