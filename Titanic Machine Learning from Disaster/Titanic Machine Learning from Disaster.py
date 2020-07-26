# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Octaves0911

##Importing Libraries and Files
"""
from zipfile import ZipFile
with ZipFile('titanic.zip','r') as zip:
    zip.printdir()
    zip.extractall()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

"""##Loading the datasets"""

train_data=pd.read_csv('train.csv')

test_data=pd.read_csv('test.csv')



"""#Feature Engineering, Cleaning and Manipulation

###Creating new columns
"""

train_data["Has_Cabin"]=train_data['Cabin'].isnull().astype(int)

train_data["Has_Cabin"]=(train_data["Has_Cabin"]==0)

train_data['Has_Cabin']=train_data["Has_Cabin"].map({True:1,False:0})

test_data["Has_Cabin"]=test_data['Cabin'].isnull().astype(int)
test_data["Has_Cabin"]=(test_data["Has_Cabin"]==0)
test_data['Has_Cabin']=test_data["Has_Cabin"].map({True:1,False:0})

test_data.head()

train_data.head()

train_data["Name_len"]=train_data["Name"].apply(len)
test_data["Name_len"]=train_data["Name"].apply(len)

train_data.head()

train_data["Family_size"]=train_data["Parch"]+train_data["SibSp"]+1
test_data["Family_size"]=test_data["Parch"]+test_data["SibSp"]+1

train_data["Is_Alone"]=(train_data["Family_size"]==1).astype(int)
test_data["Is_Alone"]=(test_data["Family_size"]==1).astype(int)

train_data.head()

train_data["Embarked"].fillna('S',inplace=True)
test_data["Embarked"].fillna('S',inplace=True)

train_data["Fare"].fillna(train_data["Fare"].median(),inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(),inplace=True)

mean_train=train_data["Age"].mean()
std_train=train_data["Age"].std()
mean_test=test_data["Age"].mean()
std_test=test_data["Age"].std()
train_data['Age']=train_data['Age'].fillna(int(np.random.uniform(low=mean_train-std_train,high=mean_train+std_train)))
test_data['Age']=test_data['Age'].fillna(int(np.random.uniform(low=mean_test-std_test,high=mean_test+std_test)))



train_data["Categorical_Fare"]=pd.qcut(train_data["Fare"],4)
train_data["Categorical_Age"]=pd.qcut(train_data["Age"],5)
test_data["Categorical_Age"]=pd.qcut(test_data["Age"],5)
test_data["Categorical_Fare"]=pd.qcut(test_data["Fare"],4)

train_data.head()

import re
def get_title(name):
    title=re.search('([a-zA-Z]+)\.',name)
    if title:
        return title.group(1)
    return ""

train_data["Title"]=train_data["Name"].apply(get_title)
test_data["Title"]=test_data["Name"].apply(get_title)

train_data["Title"].replace([ 'Don', 'Rev', 'Dr',
       'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess',
       'Jonkheer'],'Rare',inplace=True)
test_data["Title"].replace([ 'Don', 'Rev', 'Dr',
       'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess',
       'Jonkheer'],'Rare',inplace=True)

train_data["Title"].replace(['Mlle','Ms'],'Miss',inplace=True)
train_data["Title"].replace('Mme','Mrs',inplace=True)
test_data["Title"].replace(['Mlle','Ms'],'Miss',inplace=True)
test_data["Title"].replace('Mme','Mrs',inplace=True)



train_data.head()

"""#Encoding"""

train_data["Embarked"]=train_data["Embarked"].map({'S':0,"C":1,"Q":2})
test_data["Embarked"]=test_data["Embarked"].map({'S':0,"C":1,"Q":2})

train_data["Sex"]=train_data["Sex"].map({'male':0,"female":1})
test_data["Sex"]=test_data["Sex"].map({'male':0,"female":1})

train_data["Title"]=train_data["Title"].map({'Mr':1,"Mrs":2,"Miss":3,"Master":4,"Rare":5})
test_data["Title"]=test_data["Title"].map({'Mr':1,"Mrs":2,"Miss":3,"Master":4,"Rare":5})
train_data["Title"].fillna(0,inplace=True)
test_data["Title"].fillna(0,inplace=True)

train_data.loc[train_data["Fare"]<=7.91,'Fare']=0
train_data.loc[(train_data["Fare"]>7.91)&(train_data["Fare"]<=14.454),'Fare']=1
train_data.loc[(train_data["Fare"]>14.454)&(train_data["Fare"]<=31.0),'Fare']=2
train_data.loc[(train_data["Fare"]>31.0)&(train_data["Fare"]<=512.329),'Fare']=3

test_data.loc[test_data["Fare"]<=7.91,'Fare']=0
test_data.loc[(test_data["Fare"]>7.91)&(test_data["Fare"]<=14.454),'Fare']=1
test_data.loc[(test_data["Fare"]>14.454)&(test_data["Fare"]<=31.0),'Fare']=2
test_data.loc[(test_data["Fare"]>31.0),'Fare']=3

train_data.loc[train_data["Age"]<=16,"Age"]=0
train_data.loc[(train_data["Age"]>16)&(train_data["Age"]<=32),'Age']=1
train_data.loc[(train_data["Age"]>32)&(train_data["Age"]<=48),'Age']=2
train_data.loc[(train_data["Age"]>48)&(train_data["Age"]<=64),'Age']=3
train_data.loc[(train_data["Age"]>64),'Age']=4

test_data.loc[test_data["Age"]<=16,"Age"]=0
test_data.loc[(test_data["Age"]>16)&(test_data["Age"]<=32),'Age']=1
test_data.loc[(test_data["Age"]>32)&(test_data["Age"]<=48),'Age']=2
test_data.loc[(test_data["Age"]>48)&(test_data["Age"]<=64),'Age']=3
test_data.loc[(test_data["Age"]>64),'Age']=4

train_data["Fare"]=train_data["Fare"].astype(int)
train_data["Age"]=train_data["Age"].astype(int)
test_data["Fare"]=test_data["Fare"].astype(int)
test_data["Age"]=test_data["Age"].astype(int)

train_data.head()

drop_elements=['PassengerId','Name','Ticket','Cabin','SibSp','Categorical_Age','Categorical_Fare']

train_data=train_data.drop(columns=drop_elements)

test_data=test_data.drop(columns=drop_elements)


"""#Data Visualisation"""

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

colormap=plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title("Pearsons Coefficien of features",y=1.05,size=15)
sns.heatmap(train_data.corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)

#g=sns.pairplot(train_data[['Survived','Pclass','Sex','Age','Parch','Fare','Embarked']],hue='Survived',palette='seismic',diag_kind='kde',size=2.5,diag_kws=dict(shade=True),plot_kws=dict(s=10))

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier

X_train=train_data.drop(columns=['Survived'],axis=1)

Y_train=train_data['Survived']

X_test=test_data

X_test.info()

X_train.info()

#LOGISTIC REGRESSION
logistic=LogisticRegression()
logistic.fit(X_train,Y_train)
logistic_prediction=logistic.predict(X_test)
log_reg_data=pd.read_csv('test.csv')
log_reg_data.insert((log_reg_data.shape[1]),'Survived',logistic_prediction)

log_reg_data.to_csv("LogisticRegression.csv")

#Adaptive Boosting
adaboost=AdaBoostClassifier()
adaboost.fit(X_train,Y_train)
adaboost_prediction=adaboost.predict(X_test)
ada_data=pd.read_csv('test.csv')
ada_data.insert((ada_data.shape[1]),'Survived',adaboost_prediction)

ada_data.to_csv("AdaptiveBoosting.csv")

#Bagging Classifier
bagclass=BaggingClassifier()
bagclass.fit(X_train,Y_train)
bagclass_prediction=bagclass.predict(X_test)
bagclass_data=pd.read_csv('test.csv')
bagclass_data.insert((bagclass_data.shape[1]),'Survived',bagclass_prediction)

bagclass_data.to_csv("BaggingClassifier.csv")

#Random Forest
random=RandomForestClassifier(n_estimators=1000)
random.fit(X_train,Y_train)
random_prediction=random.predict(X_test)
random_data=pd.read_csv('test.csv')
random_data.insert((random_data.shape[1]),'Survived',random_prediction)

random_data.to_csv("RandomForest.csv")

#Decision Tree
dectree=DecisionTreeClassifier()
dectree.fit(X_train,Y_train)
dectree_prediction=dectree.predict(X_test)
dectree_data=pd.read_csv('test.csv')
dectree_data.insert((dectree_data.shape[1]),'Survived',dectree_prediction)

dectree_data.to_csv("DecisionTree.csv")

#Gradien Boosting
grad=GradientBoostingClassifier()
grad.fit(X_train,Y_train)
grad_prediction=grad.predict(X_test)
grad_data=pd.read_csv('test.csv')
grad_data.insert((grad_data.shape[1]),'Survived',grad_prediction)

grad_data.to_csv("GradientBoosting.csv")

#XGBoost
xgboost=XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=3,min_child_weight=4,colsample_bytree=0.6,reg_alpha=0.001,subsample=0.9)
xgboost.fit(X_train,Y_train)
xgboost_prediction=xgboost.predict(X_test)
xgboost_data=pd.read_csv('test.csv')
xgboost_data.insert((xgboost_data.shape[1]),'Survived',xgboost_prediction)

xgboost_data.to_csv("XGBoosting.csv")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, make_scorer, confusion_matrix, f1_score, fbeta_score

#Hyperparameter tuning
params_dict={'learning_rate':[0.1,0.001,0.01],
             'n_estimator':[1000,200,2500],
             'min_child_weight':[1,3,4],
             'colsample_bytree':[0.7,0.8],
             'subsample':[0.8,0.85,0.9,1],
             #'gamma':[0,0.2,0.3,0.8],
             'reg_alpha':[0.01,0.005,0.001]}
scoring={'AUC':'roc_auc','Accuracy':make_scorer(accuracy_score)}

gsearch=GridSearchCV(estimator=XGBClassifier(learning_rate=0.001,n_estimators=1000,max_depth=3,colsample_bytree=0.7,reg_alpha=0.001,subsample=0.8),param_grid=params_dict,cv=10,scoring=scoring,iid=False,refit='Accuracy',verbose=5,n_jobs=-1)

 

print(gsearch.best_params_)

print(gsearch.best_score_)

#Hypertuned XGBOOSt Accuracy 0.78468
hxgboost=XGBClassifier(learning_rate=0.001,n_estimators=1000,max_depth=3,colsample_bytree=0.7,reg_alpha=0.001,subsample=0.8)
hxgboost.fit(X_train,Y_train)
hxgboost_prediction=hxgboost.predict(X_test)
hxgboost_data=pd.read_csv('test.csv')
hxgboost_data.insert((hxgboost_data.shape[1]),'Survived',hxgboost_prediction)
hxgboost_data.to_csv("Hyperturnd XGBoosting.csv")
