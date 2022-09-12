
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Data:
    
    def __init__(self,path,target):
        self.df = pd.read_csv(path,delimiter=";")
        self.encode()
        self.y = self.df[target].values.reshape(-1,1)
        self.df.drop([target],axis=1,inplace=True)
        
    def encode(self):
        #feature types
        features = self.__feature_categorization(self.df)
        #categorical > labelencoding
        for f in features["Categorical"]:
            self.label_encoding(f)
        #numerical
        for f in features["Numerical"]:
            self.standard_scaling(f)
            
    def get_df(self):
        return self.df, self.y

    def check_missing_values(self):
        print(self.df.isnull().sum())
        return True
    
    def label_encoding(self,col):
        self.df[col] = self.df[col].astype(str)
        le = LabelEncoder()
        x = self.df[col].values.reshape(-1,1)
        le.fit(x)
        x_label_encoded = le.transform(x).reshape(-1,1)
        self.df[col] = x_label_encoded.reshape(len(x_label_encoded),1)
        return True
    
    def standard_scaling(self,col):
        x = self.df[col].values.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(x)
        self.df[col] = scaler.transform(x)
        return True
    
    def __feature_categorization(self,df):
        numerical = df.select_dtypes(include=[np.number]).columns.values.tolist()
        categorical = df.select_dtypes(include=['object']).columns.values.tolist()
        datetimes = df.select_dtypes(include=['datetime',np.datetime64]).columns.values.tolist()
        features = {"Numerical":numerical,"Categorical":categorical,"Datetimes":datetimes}
        return features 
path = PATH
D = Data(path,"y")
df, y = D.get_df()

class LogisticRegression:
    
    def __init__(self,epoch=50000,lr=0.001,method="gradient"):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.b = None
        self.method = method
        self.cost_list = []
        
    def fit(self,X,y):
        #initialization
        X, self.w = self.__initialization(X)
        #iterations
        for i in range(self.epoch):
            #prediction
            z = self.__get_z(X)
            #sigmoid
            a = self.__sigmoid(z)
            #calculate lost
            #gradient
            if self.method == "gradient":
                cost = self.__calculate_loss(a, y)    
                self.cost_list.append(cost)
                gr = self.__gradient_descent(X, a, y)
            elif self.method == "likelihood":
                l = self.__log_likelihood(X,y)
                self.cost_list.append(l)
                gr = self.__gradient_l(X, a, y)
            #update w
            self.w = self.__update(self.w, gr)
        self.b = X[:,0]
        return True
    
    def predict(self,X):
        b = np.ones((X.shape[0],1))
        X = np.concatenate((b,X),axis=1)
        z = self.__get_z(X)
        s = self.__sigmoid(z)
        result = [1 if x >=0.5 else 0 for x in s]
        return result
    
    def get_cost(self):
        return self.cost_list
    
    def __get_z(self,X):
        return np.dot(X,self.w) 

    def __sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def __gradient_descent(self,X,a,y):
        return np.dot(X.T,(a-y)) / y.shape[0]
    
    def __update(self,w,gr):
        return w - self.lr * gr
    
    def __calculate_loss(self,a,y):
        return (-y * np.log(a) - (1-y) * np.log(1-a)).mean()
            
    def __initialization(self,X):
        b = np.ones((X.shape[0],1))
        X = np.concatenate((b,X),axis=1)
        w = np.zeros(X.shape[1]).reshape(-1,1)
        return X,w
    
    def __log_likelihood(self,X,y):
        z = np.dot(X,self.w)
        l = np.sum(y*z - np.log(1+np.exp(z)))
        return l
    
    def __gradient_l(self,X,a,y):
        return np.dot(X.T,y-a)