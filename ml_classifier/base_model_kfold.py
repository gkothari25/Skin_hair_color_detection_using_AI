import os
import cv2
import glob
import pickle
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.svm import SVC
import matplotlib.pyplot  as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


#58 black 0
#69 blonde 2
#61 grey 1 

#load csv 
df = pd.read_csv('hist_hair.csv')
df1 = df.copy()

Y = df1.iloc[:]['5']
#X = df1.loc[:,['16','17']]
#print(X)
X = df1.drop(['5'], axis = 1)

#print(X.shape,Y.shape)
#print(df.shape)

scaler = StandardScaler()
scaler.fit(X)

filename1 = 'scaler_final.pkl'
pickle.dump(scaler, open(filename1, 'wb'))

X_norm = scaler.transform(X)

#print(X_norm['5'])
#print(X_norm.shape)

#X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, random_state = 0,shuffle = True)

#apply k fold in them

kf = KFold(n_splits=10)
a = kf.split(X_norm,Y)
c = 1

for train_index, test_index in a:
    x_train,x_test = X_norm[train_index],X_norm[test_index]
    
    y_train,y_test = Y[train_index],Y[test_index]

    clf = SVC(C=100,decision_function_shape='ovo', degree=3, gamma=0.1, kernel='rbf',probability=True)
    #clf = RandomForestClassifier(n_estimators=180, criterion='gini')
    clf.fit(x_train,y_train)
    file_name = 'model{}'.format(c)
    #pickle.dump(clf,open(file_name,'wb'))
    c= c+1
    pred = clf.predict(x_test)
    s = accuracy_score(y_test,pred)
    s1= classification_report(y_test,pred)
    #print(X_test.shape)
    #print(pred.shape)

    print("accuracy score is \n",s)
    print('classification report is---')
    print(s1)