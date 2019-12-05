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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

#load csv 
df = pd.read_csv('hair_data_new.csv')
df1 = df.copy()

Y = df1.iloc[:]['5']
#X = df1.loc[:,['16','17']]
#print(X)
X = df1.drop(['5'],axis = 1)

#print(X.shape,Y.shape)
#print(df.shape)

scaler = StandardScaler()
scaler.fit(X)

filename1 = 'scaler_final_new.pkl'
pickle.dump(scaler, open(filename1, 'wb'))

X_norm = scaler.transform(X)


#boruta applied 
#rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
#feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
#feat_selector.fit(X_norm, Y)

#print(X_norm['5'])
#print(X_norm.shape)

#pca applied
pca = PCA(n_components=4)
pca.fit(X_norm,Y)



X_train, X_test, y_train, y_test = train_test_split(X_norm, Y,test_size=0.10,shuffle = True)

clf = SVC(C=1000,decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',probability=True)
#clf = RandomForestClassifier(n_estimators=180, criterion='gini')
X_trainp = pca.transform(X_train) 
X_testp  = pca.transform(X_test)

clf.fit(X_trainp,y_train)

file_name2 = 'split_model_new.pkl'
pickle.dump(clf,open(file_name2,'wb'))
pred = clf.predict(X_testp)
s = accuracy_score(y_test,pred)
s1= classification_report(y_test,pred)
cm = confusion_matrix(y_test,pred)
#print(X_test.shape)
#print(pred.shape)

print("accuracy score is \n",s)
print('classification report is---')
print(s1)
print(cm)