import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
# split a dataset into train and test sets


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(
            self.args['fileName'])  # , nrows=100000

    def split_train_test(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)
        return X_train, X_test, y_train, y_test
    # get dataframe and return numpy arrey and columns names and range list

    def hist(self):
        names = self.data["name"]
        states = self.data["state"]

        di = {}
        stops = stopwords.words()
        for i, name in enumerate(names):
            if(states[i] == "successful"):
                state = 1
            else:
                state = 0
            ar = name.split(" ")
            for x in ar:
                if(x.lower() not in stops):
                    x = x.lower()
                    if(x in di.keys()):
                        di[x] = (di[x][0]+1, di[x][1]+state,
                                 (di[x][1]+state)/(di[x][0]+1))
                    else:
                        di[x] = (1, state, state/1)
        sorted_dict = dict(
            sorted(di.items(), key=lambda item: item[1][0], reverse=True))
        return sorted_dict

    def preprocess(self):

        self.data = self.data.loc[self.data['state'].isin(
            ['failed', 'successful'])]
        # drop na values
        self.data = self.data.dropna()
        self.data = self.data.drop(['ID',
                                    'launched', 'cat_sub_cat'], 1)

        cat_columns = ['state']
        self.data['state'] = self.data['state'].astype('category')
        self.data[cat_columns] = self.data[cat_columns].apply(
            lambda x: x.cat.codes)

        category = pd.get_dummies(self.data['category'], drop_first=True)
        main_category = pd.get_dummies(
            self.data['main_category'], drop_first=True)
        currency = pd.get_dummies(self.data['currency'], drop_first=True)
        country = pd.get_dummies(self.data['country'], drop_first=True)
        word_frequencies_dict = self.hist()
        s = {k: word_frequencies_dict[k]
             for k in list(word_frequencies_dict)[:250]}
        s_keys = s.keys()
        for key in s_keys:
            self.data[key] = 0
        for idx, name in enumerate(self.data["name"]):
            names = name.split(" ")
            for n in names:
                if(n.lower() in s_keys):
                    self.data.set_value(idx, n.lower(), 1)
                    #self.data.iloc[idx][n.lower()] = 1
        self.data = pd.concat(
            [self.data, category, main_category, currency, country], axis=1)
        self.data = self.data.drop(
            ['category', 'main_category', 'currency', 'country'], 1)
        Y = self.data.loc[:, 'state'].values
        self.data = self.data.drop(['state', 'name'], 1)
        columns = self.data.columns

        """Scaler1 = StandardScaler()
        self.data = pd.DataFrame(Scaler1.fit_transform(self.data))"""

        self.data.columns = columns
        features = self.data.iloc[:, 2:].columns.tolist()
        X = self.data.iloc[:, 2:].values
        Scaler1 = StandardScaler()
        X = Scaler1.fit_transform(X)
        return X, Y, features
        # balance data:
        """g=self.data.groupby('state')
        self.data=g.apply(lambda x: x.sample(
            g.size().min()).reset_index(drop=True))
        self.data[cat_columns] = self.data[cat_columns].apply(
            lambda x: x.cat.codes)
        """
