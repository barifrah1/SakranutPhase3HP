import random


class HyperParameters():
    hyper_p = ['batch_size', 'n_epochs', 'lr', 'weight_decay', 'dropout_param']
    hyperparams_dict = {'batch_size': 0, 'n_epochs': 1, 'lr': 2,
                        'weight_decay': 3, 'dropout_param': 4}

    def __init__(self, hyper_params=()):
        # batch_size=None, n_epochs=None, lr=None, weight_decay=None, number_of_words_features=None, dropout_param
        self.optional_values = {}
        self.hyper_params = ()
        if(hyper_params == ()):
            self._setHyperParametersOptionalValuse()
            hp_list = []
            for i in range(len(HyperParameters.hyper_p)):
                p = HyperParameters.hyper_p[i]
                hp_list.append(
                    random.choice(self.optional_values[p]))
            self.hyper_params = tuple(hp_list)
        else:
            self.hyper_params = hyper_params
            self._setHyperParametersOptionalValuse()

    def _setHyperParametersOptionalValuse(self):
        self.optional_values['n_epochs'] = [250, 200, 150, 100, 10]
        self.optional_values['batch_size'] = [30000, 20000, 10000, 5000]
        self.optional_values['lr'] = [1, 1e-1, 1e-2, 1e-4, 1e-6]
        self.optional_values['weight_decay'] = [1, 1e-2, 1e-3, 1e-5, 1e-8]
        self.optional_values['dropout_param'] = [0.8, 0.5, 0.3]

    def get(self):
        return self.hyper_params

    def set(self, hp: tuple):
        self.hyper_params = hp

    def getOptionalActions(self):
        possible_actions = {}
        for ind, f in enumerate(HyperParameters.hyper_p):
            if(self.hyper_params[ind] == self.optional_values[f][0]):
                possible_actions[f] = {'D': 1}
            elif(self.hyper_params[ind] == self.optional_values[f][-1]):
                possible_actions[f] = {'U': 1}
            else:
                possible_actions[f] = {'U': 1, 'D': 1}
        possible_actions['n_epochs']['S'] = 1
        return possible_actions
