# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:45:01 2019

@author: puneet pandey
"""

#DATA PREPROCESSING
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset
 df=pd.read_csv('Data.csv')
X=df.iloc[:,:-1].values #matrix of independent variable. or matrix of feature variable.
Y=df.iloc[:,3].values  #matrix of dependent variable.

#taking care of missing data
#use sklearn or pandas (we will use sklearn here)
from sklearn.preprocessing import Imputer  #importing the class imputer from sklearn famliy
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0) #making object of class Imputer .(press ctrl+i to see documentation)
imputer=imputer.fit(X[:,1:3]) #to fit imputer of change to data. or column
X[:,1:3]=imputer.transform(X[:,1:3]) #to transform the data from the changes that we have made.

#catogorical variable
#label encoding for catagorical data .
from sklearn.preprocessing import LabelEncoder #importing the library and class used for encoding
labelencoder_X=LabelEncoder() #creating  a object of class 
X[:,0]=labelencoder_X.fit_transform(X[:,0]) #using the method and saving the data to same column .
from sklearn.preprocessing import OneHotEncoder #onehot encoding of same data 
onehotencoding_x=OneHotEncoder(categorical_features=[0])  ##object of inehot encoder 
X=onehotencoding_x.fit_transform(X).toarray()##transforming the data and fittig the data in X only saving in array form.

labelencoder_y=LabelEncoder() #encoding the Y column of data set using label encoding
Y=labelencoder_y.fit_transform(Y)


# test train and split
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling on dataset
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#we can also do feature scaling on dataset y (dependent value).