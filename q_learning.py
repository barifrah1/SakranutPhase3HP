import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from matplotlib import pyplot as plt
from tqdm import tqdm
from hyperparameters import HyperParameters
from NN import Net, train, predict
from random import random, choice, shuffle
import pickle
from copy import deepcopy


class Q_Learning():

    # in case building the net for the final model
    def __init__(self, hyper_params_to_start,  args, learner, X_train=None, y_train=None, X_val=None, y_val=None, Q=None):
        if(Q != None):
            self.Q_func = Q
        self.Q_func = {}
        self.hyper_params = hyper_params_to_start
        self.nextIterHp = []
        self.args = args
        self.learner = learner
        self.original_learner = learner
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.hyperparams_dict = {'batch_size': 0, 'n_epochs': 1, 'lr': 2,
                                 'weight_decay': 3, 'dropout_param': 4}
        self.policy = {}

    def getQvalue(self, state, action):
        return self.Q_func[state][action]

    def setQvalue(self, state, action, value):
        self.Q_func[state][action] = value

    def getNextState(self, hp, action):
        state = hp.get()
        next_state = list(state)
        feature_index = self.hyperparams_dict[action[0]]
        options_vector = hp.optional_values[action[0]]
        curren_val_index = options_vector.index(state[feature_index])
        if(action[1] == 'U'):
            next_state[feature_index] = options_vector[curren_val_index-1]
        elif(action[1] == 'D'):
            next_state[feature_index] = options_vector[curren_val_index+1]
        else:
            return state
        return tuple(next_state)

    def _bestCurrentAction(self, hp, random=False):
        state = hp.get()
        if(state not in self.Q_func.keys()):
            self._addStateAndPossibleActionsToQ(hp)
        actions_vector = list(self.Q_func[state].keys())
        shuffle(actions_vector)
        if(random == True):
            return choice(list(actions_vector))
        max_value = -10000000000000000000000000000000000000
        max_action = -1
        for action in actions_vector:
            if(self.getQvalue(state, action) > max_value):
                max_value = self.getQvalue(state, action)
                max_action = action
        if(max_action != -1):
            return max_action
        else:
            raise Exception(
                f"Sorry,no appropriate action was found for state {state}")

    def chooseBestAction(self, hp):
        state = hp.get()
        actions_vector = self.Q_func[state].keys()
        max_value = -10000000000000000000000000000000000000
        max_action = -1
        for action in actions_vector:
            next_state = self.getNextState(hp, action)
            if(next_state not in self.Q_func.keys()):
                new_p = HyperParameters(next_state)
                self._addStateAndPossibleActionsToQ(new_p)

            if(self.getQvalue(next_state, action) > max_value):
                max_value = self.getQvalue(next_state, action)
                max_action = action
                save_next_state = next_state
        if(max_action != -1):
            f2 = open("state_trace.txt", "a")
            f2.write(str(state)+" "+str(max_action)+" "+max_value+"\n")
            f2.close()
            return max_action, save_next_state
        else:
            raise Exception(
                f"Sorry,no appropriate action was found for state {state}")

    """def _chooseRandomAction(self, state):
        actions_vector = self.Q_func[state].keys()
        chosenRandomAction = random.choice(list(actions_vector))
        next_state = self.getNextState(state, chosenRandomAction)
        if(next_state not in self.Q_func.keys()):
            self._addStateAndPossibleActionsToQ(HyperParameters(next_state))
        return chosenRandomAction, next_state"""

    def _reward(self, val_loss, epoch):
        epsilon = (100*np.exp(-val_loss) - 100*np.exp(-0.62))
        if(val_loss < 0.595):
            return 10000*(0.595-val_loss)
        return epsilon

    def _addStateAndPossibleActionsToQ(self, hp):
        state = hp.get()

        if(state not in self.Q_func.keys()):
            self.Q_func[state] = {}
            init = hp.getOptionalActions()
            for f in init.keys():
                for a in init[f].keys():
                    self.setQvalue(state, (f, a), 1)

    def _trainLearnerAndGetPreds(self, hyper_params, iter=None):
        val_loss = train(self.X_train, self.y_train, self.learner, self.X_val,
                         self.y_val, hyper_params.get(), criterion=nn.BCELoss())
        return val_loss

    def updateQfile(self, reward):
        with open('Q.pickle', 'wb') as handle:
            pickle.dump(self.Q_func, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open('reward.pickle', 'wb') as handle:
            pickle.dump(reward, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def calcPolicy(self):
        for state in self.Q_func.keys():
            actions_vector = self.Q_func[state].keys()
            max_value = -10000000000000000000000000000000000000
            max_action = -1
            for action in actions_vector:
                if(self.getQvalue(state, action) > max_value):
                    max_value = self.getQvalue(state, action)
                    max_action = action
            self.policy[state] = max_action

    def exploitPolicy(self):
        total_val_loss = 0
        reward_by_episode = []
        for i in range(10):
            p = HyperParameters(())
            state = p.get()
            total_val_loss = 0
            for iter in range(20):
                if(state in self.policy.keys()):
                    action = self.policy[state]
                    print(state, action, 1)
                else:
                    actions_vec = []
                    actions = p.getOptionalActions()
                    for f in actions.keys():
                        for a in actions[f].keys():
                            actions_vec.append((f, a))
                    action = choice(actions_vec)
                    print(state, action, 2)
                next_state = self.getNextState(p, action)
                new_p = HyperParameters(next_state)
                val_loss, epoch = self._trainLearnerAndGetPreds(new_p)
                total_val_loss += val_loss
                p = new_p
            reward_by_episode.append(total_val_loss/20)
        return sum(reward_by_episode)/10

    def q_learning_loop(self, feature_num, is_random_policy=False):
        reward_by_episode = []
        val_by_ep = []
        for episode in tqdm(range(self.args.num_episodes)):
            """trained_val_auc = -1
            auc_validation = []
            trained_val_loss = -1
            validation_loss_list = []"""
            p = HyperParameters(())
            acc_reward = 0
            val_loss = 0
            r = 0
            for iter in range(self.args.num_of_iters):
                self.learner = Net(feature_num, dropout=p.get()[4])
                # here we should train the learner_net
                state = p.get()
                sample = random()
                if(sample < self.args.getEpsilon(iter)):
                    action = self._bestCurrentAction(p, random=True)
                else:
                    action = self._bestCurrentAction(p)
                    if(self.getQvalue(state, action) < self.args.threshold):
                        reward_by_episode.append(
                            acc_reward/(iter+1))
                        print(
                            f"episode {episode+1}  iters:{iter+1} : val_loss: {val_loss}  reward: {r} ")
                        self.updateQfile(reward_by_episode)
                        break
                next_state = self.getNextState(p, action)
                new_p = HyperParameters(next_state)
                val_loss, epoch = self._trainLearnerAndGetPreds(new_p)
                r = self._reward(val_loss, epoch)
                acc_reward += r
                # case when we havent visit in this state yet, then defint its q value to uniform distribution
                if(next_state not in self.Q_func.keys()):
                    self._addStateAndPossibleActionsToQ(p)
                bestAction = self._bestCurrentAction(new_p)
                td_error = r + self.args.gamma * self.getQvalue(next_state,
                                                                bestAction) - self.getQvalue(state, action)
                self.setQvalue(state, action, self.getQvalue(
                    state, action)+self.args.getEta(iter)*td_error)
                f2 = open("state_trace.txt", "a")
                f2.write(str(state)+" "+str(action)+" " +
                         str(self.getQvalue(state, action))+"\n")
                f2.close()
                p = new_p

                """if iter % 50 == 0:
                    print(
                        f"Current validation loss on epoch {iter+1} is: {trained_val_loss} ")"""
            reward_by_episode.append(acc_reward/self.args.num_of_iters)
            print(
                f"episode {episode+1} : avg_reward: {acc_reward/self.args.num_of_iters}  ")
            self.updateQfile(reward_by_episode)
            if(episode % 10 == 0):
                self.calcPolicy()
                val_loss_be_episode = self.exploitPolicy()
                val_by_ep.append(val_loss_be_episode)
                with open('val_loss_by_episode_policy.pickle', 'wb') as handle:
                    pickle.dump(val_by_ep, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                f3 = open("improvment_trace.txt", "a")
                f3.write(str(episode)+" "+str(val_loss_be_episode)+" " +
                         str(val_by_ep)+"\n")
                f3.close()
        return reward_by_episode
