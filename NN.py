import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from data_loader import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from matplotlib import pyplot as plt





class Net(nn.Module):
    
    def __init__(self,feature_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_num, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 2)
        #self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        return x
    
def predict(x,y,model,auc_list):
    with torch.no_grad():
        x_var = Variable(torch.FloatTensor(x)) 
        test_result = model(x_var)
        #loss = criterion(test_result,y_var)
        values, labels = torch.max(test_result, 1)
        auc=roc_auc_score(y,labels)
        auc_list.append(auc)
    return auc,auc_list
def train(X_train,y_train,model,x_test,y_test,
          batch_size,
          n_epochs,
          criterion = nn.CrossEntropyLoss(),
):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    batch_no = len(X_train) // batch_size
    #print(batch_no)
    train_loss = 0
    auc_test=[]
    #train_loss_min = np.Inf
    for epoch in range(n_epochs):
        for i in range(batch_no):
            start = i*batch_size
            end = start+batch_size
            x_var = Variable(torch.FloatTensor(X_train[start:end]))
            y_var = Variable(torch.LongTensor(y_train[start:end])) 
            optimizer.zero_grad()
            output = model(x_var)
            loss = criterion(output,y_var)
            loss.backward()
            optimizer.step()
            
            values, labels = torch.max(output, 1)
            num_right = np.sum(labels.data.numpy() == y_train[start:end])
            train_loss += loss.item()*batch_size
        
        train_loss = train_loss / len(X_train)
        #print('labels',labels)
        #print('y_train',y_train[start:end])
        with torch.no_grad():
            x_var = Variable(torch.FloatTensor(X_train))
            output_epoch=model(x_var)
            values, labels=torch.max(output_epoch, 1)
            auc=roc_auc_score(y_train,labels)
        print('auc',auc)
        """if train_loss <= train_loss_min:
            print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))
            torch.save(model.state_dict(), "model.pt")
            train_loss_min = train_loss
        """
        
        if epoch:
            print('')
            print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {} \tTrain auc:".format(epoch+1, train_loss,num_right / len(y_train[start:end]),auc ))
        print('Training Ended! ')
        if epoch%10==0:
            pre,auc_test=predict(x_test,y_test,model,auc_test)
            print(f'predict auc in epoec {epoch+1} is {pre}')
    plt.plot(auc_test)
    plt.show()
            

        


