#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('WineQT.csv')
print(df.head())


# In[3]:


df.info()


# In[4]:


df.describe().T


# In[5]:


df.isnull().sum()


# In[7]:


for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
        df.isnull().sum().sum()


# In[8]:


df.hist(bins=20, figsize=(10, 10))
plt.show()


# In[9]:


plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[10]:


plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


# In[11]:


df = df.drop('total sulfur dioxide', axis=1)


# In[12]:


df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


# In[13]:


df.replace({'white': 1, 'red': 0}, inplace=True)


# In[14]:


features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)

xtrain.shape, xtest.shape


# In[15]:


norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# In[16]:


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
    models[i].fit(xtrain, ytrain)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        ytest, models[i].predict(xtest)))
    print()


# In[17]:


metrics.plot_confusion_matrix(models[1], xtest, ytest)
plt.show()


# In[18]:


print(metrics.classification_report(ytest,
                                    models[1].predict(xtest)))

