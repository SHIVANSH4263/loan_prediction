#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Importing Libraries and Dataset
# Firstly we have to import libraries : 
#Pandas – To load the Dataframe
#Matplotlib – To visualize the data features i.e. barplot
#Seaborn – To see the correlation between features using heatmap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("D:\LOAN PROJECT\LoanApprovalPrediction.csv")
#Once we imported the dataset, let’s view it using the below command.
data.head(5)


# In[8]:



obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))


# In[9]:


#As Loan_ID is completely unique and not correlated with any of the other column, So we will drop it using .drop() function.
# Dropping Loan_ID column
data.drop(['Loan_ID'],axis=1,inplace=True)


# In[11]:


#Visualize all the unique values in columns using barplot. This will simply show which value is dominating as per our dataset.
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18,36))
index = 1

for col in object_cols:
  y = data[col].value_counts()
  plt.subplot(11,4,index)
  plt.xticks(rotation=90)
  sns.barplot(x=list(y.index), y=y)
  index +=1


# In[12]:


#As all the categorical values are binary so we can use Label Encoder for all such columns and the values will change into int datatype.
# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how
# to understand word labels.
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
  data[col] = label_encoder.fit_transform(data[col])

# To find the number of columns with
# datatype==object
obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))


# In[13]:


plt.figure(figsize=(12,6))

sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f',
            linewidths=2,annot=True)


# In[14]:


#Now we will use Catplot to visualize the plot for the Gender, and Marital Status of the applicant.
sns.catplot(x="Gender", y="Married",
hue="Loan_Status",
kind="bar",
data=data)


# In[15]:


#Now we will find out if there is any missing values in the dataset using below code.

for col in data.columns:
  data[col] = data[col].fillna(data[col].mean()) 
    
data.isna().sum()


# In[16]:


#model training
from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']
X.shape,Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
test_size=0.4,random_state=1)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[18]:


#model training and evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators = 7,criterion = 'entropy',random_state =7)
svc = SVC()
lc = LogisticRegression()

# making predictions on the training set
for clf in (rfc, knn, svc,lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print("Accuracy score of ",
        clf.__class__.__name__,
        "=",100*metrics.accuracy_score(Y_train,Y_pred))


# In[19]:


#prediction on the test set
# making predictions on the testing set
for clf in (rfc, knn, svc,lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("Accuracy score of ",
        clf.__class__.__name__,"=",
        100*metrics.accuracy_score(Y_test,Y_pred))


# In[ ]:




