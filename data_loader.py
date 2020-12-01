import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# split a dataset into train and test sets



class DataLoader:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args['fileName'], nrows=100000)

    def split_train_test(self,x,y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        return X_train,X_test,y_train,y_test
    # get dataframe and return numpy arrey and columns names and range list

    def preprocess(self):
        
        self.data = self.data.loc[self.data['state'].isin(
            ['failed', 'successful'])] 
        # drop na values   
        self.data = self.data.dropna()   
        self.data = self.data.drop(['ID','goal','pledged','usd pledged',
                  'name','deadline','launched','cat_sub_cat','backers','usd_pledged_real','backers_s','pledged_s','mean_pl','mean_bac'],1)
        
        cat_columns=['state']
        self.data['state'] = self.data['state'].astype('category')    
        self.data[cat_columns] = self.data[cat_columns].apply(lambda x: x.cat.codes)
        
        category=pd.get_dummies(self.data['category'],drop_first=True)
        main_category=pd.get_dummies(self.data['main_category'],drop_first=True)
        currency=pd.get_dummies(self.data['currency'],drop_first=True)
        country=pd.get_dummies(self.data['country'],drop_first=True)
        
        self.data = pd.concat([self.data,category,main_category,currency,country],axis=1)
        self.data = self.data.drop(['category','main_category','currency','country'],1)
        
        Y = self.data.loc[:, 'state'].values
        self.data = self.data.drop(['state'],1)
        columns = self.data.columns
        Scaler1 = StandardScaler()
        self.data = pd.DataFrame(Scaler1.fit_transform(self.data))
        
        self.data.columns=columns
        features = self.data.iloc[:,2:].columns.tolist()   
        X = self.data.iloc[:,2:].values


        return X,Y,features
        #balance data:
        """g=self.data.groupby('state')
        self.data=g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))   
        self.data[cat_columns] = self.data[cat_columns].apply(lambda x: x.cat.codes)
        """

