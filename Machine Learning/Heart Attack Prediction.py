#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv("heartat.csv")


# In[4]:


data


# In[5]:


#Understanding each variable
data.info()


# In[6]:


#See a statistical summary of variables with numeric types
data.describe()


# In[7]:


data.shape


# In[8]:


#Drop duplicate value
data=data.drop_duplicates()


# In[9]:


data.shape


# In[10]:


#Check the missing value
data.isnull().sum()


# In[11]:


data


# In[13]:


from sklearn.model_selection import train_test_split

##partition data into data training and data testing
train,test = train_test_split(data,test_size = 0.20 ,random_state = 111)
    
##seperating dependent and independent variables on training and testing data
train_X = train.drop(labels='output',axis=1)
train_Y = train['output']
test_X  = test.drop(labels='output',axis=1)
test_Y  = test['output']


# In[14]:


#Proportion before smote
train_Y.value_counts()


# In[16]:


from imblearn.over_sampling import SMOTE

#handle imbalance class using oversampling minority class with smote method
os = SMOTE(sampling_strategy='minority',random_state = 123,k_neighbors=5)
train_smote_X,train_smote_Y = os.fit_resample(train_X,train_Y)
train_smote_X = pd.DataFrame(data = train_smote_X,columns=train_X.columns)
train_smote_Y = pd.DataFrame(data = train_smote_Y)


# In[17]:


#Proportion after smote
train_smote_Y.value_counts()


# # SVM

# In[18]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm=SVC()
param_grid = { 
    "C" : [0.1, 1],
    "gamma" : [0.1, 1],
    "kernel":["linear","rbf"]
}


# In[20]:


from sklearn.model_selection import GridSearchCV
CV_svm = GridSearchCV(estimator=svm, param_grid=param_grid, cv= 2)
CV_svm.fit(train_smote_X, train_smote_Y)


# In[23]:


#evaluation
pred=CV_svm.predict(test_X)


# In[24]:


from sklearn.metrics import accuracy_score
print("Accuracy for Logreg on test data: ",accuracy_score(test_Y,pred))


# In[25]:


from sklearn.metrics import confusion_matrix
CF=confusion_matrix(test_Y, pred)
CF


# In[26]:


from sklearn.metrics import classification_report
target_names = ['No','Yes']
print(classification_report(test_Y,pred, target_names=target_names))


# # Regresi Logistik

# In[48]:


import statsmodels.api as sm 
exog = sm.add_constant(train_smote_X)
log_reg = sm.Logit(train_smote_Y, exog).fit() 
print(log_reg.summary())


# In[50]:


exog = sm.add_constant(train_smote_X.drop(labels="fbs",axis=1))
log_reg = sm.Logit(train_smote_Y, exog).fit() 
print(log_reg.summary())


# In[51]:


pred=log_reg.predict(sm.add_constant(test_X.drop(labels="fbs",axis=1)))


# In[52]:


from sklearn.metrics import accuracy_score
print("Accuracy for Logreg on test data: ",accuracy_score(test_Y,np.round(pred)))


# In[53]:


from sklearn.metrics import confusion_matrix
CF=confusion_matrix(test_Y, np.round(pred))
CF


# In[54]:


from sklearn.metrics import classification_report
target_names = ['No','Yes']
print(classification_report(test_Y, np.round(pred), target_names=target_names))


# In[ ]:




