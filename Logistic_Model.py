#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
import pickle
warnings.filterwarnings("ignore")


# In[68]:


data = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")


# In[69]:


data


# In[70]:


df


# In[71]:


data.shape


# In[72]:


df.shape


# In[73]:


data.head()


# In[74]:


data.info()


# In[75]:


data.describe()


# In[76]:



data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].mean())


# In[77]:


data.describe()


# In[78]:


data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].median())


# In[79]:


data.duplicated().any()


# In[80]:



plt.figure(figsize=(8,6))
sns.countplot(df['Loan_Status']);

print('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))
print('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))


# In[81]:


df.columns


# In[82]:


grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Credit_History');


# In[83]:



grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Gender');


# In[84]:


plt.figure(figsize=(15,5))
sns.countplot(x='Married', hue='Loan_Status', data=df);


# In[85]:



plt.figure(figsize=(15,5))
sns.countplot(x='Dependents', hue='Loan_Status', data=df);


# In[86]:


grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Education');


# In[87]:



grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Self_Employed');


# In[88]:



plt.figure(figsize=(15,5))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df);


# In[89]:


plt.scatter(df['ApplicantIncome'], df['Loan_Status']);


# In[90]:


df.groupby('Loan_Status').median()


# In[91]:


df.isnull().sum().sort_values(ascending=False)


# In[92]:


df.dropna(inplace=True)


# In[93]:


df['Loan_Status'].replace('N',0,inplace=True)
df['Loan_Status'].replace('Y',1,inplace=True)


# In[94]:


df2=df.drop(labels=['ApplicantIncome'],axis=1)
df2=df2.drop(labels=['CoapplicantIncome'],axis=1)
df2=df2.drop(labels=['LoanAmount'],axis=1)
df2=df2.drop(labels=['Loan_Amount_Term'],axis=1)
df2=df2.drop(labels=['Loan_ID'],axis=1)


# In[95]:


le=LabelEncoder()


# In[96]:


ohe=OneHotEncoder()


# In[97]:


df2['Property_Area']=le.fit_transform(df2['Property_Area'])


# In[98]:


df2['Dependents']=le.fit_transform(df2['Dependents'])


# In[99]:


df2=pd.get_dummies(df2)


# In[100]:


df2=df2.drop(labels=['Gender_Female'],axis=1)
df2=df2.drop(labels=['Married_No'],axis=1)
df2=df2.drop(labels=['Education_Not Graduate'],axis=1)
df2=df2.drop(labels=['Self_Employed_No'],axis=1)
df2=df2.drop('Self_Employed_Yes',1)
df2=df2.drop('Dependents',1)
df2=df2.drop('Education_Graduate',1)


# In[101]:


X=df2.drop('Loan_Status',1)
Y=df2['Loan_Status']


# In[102]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=6)


# In[103]:


log=LogisticRegression()


# In[104]:


log.fit(x_train,y_train)
x1 = log.predict(x_test)
print(accuracy_score(y_test,x1))


# In[105]:


print(x_train)


# In[108]:


pickle.dump(log,open('Logistic_Model.pkl','wb'))


# In[109]:


model=pickle.load(open('Logistic_Model','rb'))


# In[ ]:




