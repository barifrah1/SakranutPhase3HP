from data_loader import DataLoader
from utils import args, loss_function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import torch.nn as nn
from gridsearch import GridSearch
import os
import NN
from NN import Net
from args import QArgs
from q_learning import Q_Learning
from utils import plot_loss_graph
from hyperparameters import HyperParameters
import pickle

GRID_SEARCH_MODE = False
if __name__ == '__main__':
    if(GRID_SEARCH_MODE):
        gridsearch = GridSearch()
        gridsearch.exectue_grid_search(10000)
    else:
        data_loader = DataLoader(args, False)  # False for is_grid_search mode
        # preprocessing
        X, Y, columnsInfo = data_loader.preprocess()
        # split data to train and test
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_train_validation_test(
            X, Y)
        q_args = QArgs()
        feature_num = len(columnsInfo)
        model = NN.Net(feature_num)
        hyper_p = HyperParameters()
        if(os.path.isfile('Q.pickle')) == True:
            with open('Q.pickle', 'rb') as handle:
                Q = pickle.load(handle)
            q_learn = Q_Learning(hyper_p, q_args, model,
                                 X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, Q=Q)
        else:
            q_learn = Q_Learning(hyper_p, q_args, model,
                                 X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, Q=None)
        reward_by_episode = q_learn.q_learning_loop(feature_num)
        plot_loss_graph(reward_by_episode)
