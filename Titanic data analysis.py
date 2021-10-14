#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from  matplotlib import pyplot as plt 


# train_data=pd.read_csv('train_data.csv')

# In[2]:


train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')


# In[3]:


train_data.head()

train_data.tail()
# In[4]:


train_data.tail()


# In[5]:


train_data.describe()


# In[6]:


train_data['Sex'].value_counts()


# In[7]:


plt.figure(figsize=[5,5])
plt.bar(list(train_data['Sex'].value_counts().keys()),list(train_data['Sex'].value_counts()),color=['r','g'])


# In[8]:


plt.figure(figsize=[5,5])
plt.bar(list(train_data['Survived'].value_counts().keys()),list(train_data['Survived'].value_counts()),color=['r','g'])


# In[9]:


plt.figure(figsize=[5,7])
plt.hist(train_data['Age'])
plt.title("distribution of age")
plt.show()


# In[10]:


train_data['Age'].isnull()


# In[11]:


train_data=train_data.dropna()


# In[12]:


train_data.tail()


# In[13]:


x_train=train_data[['Age']]
y_train=train_data[['Survived']]


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


dtc= DecisionTreeClassifier()


# In[16]:


dtc.fit(x_train,y_train)


# In[17]:


test_data=test_data.dropna()


# In[18]:


y=test_data[['Age']]
y_pred=dtc.predict(y)
y_pred


# In[19]:


test_data.head()


# In[ ]:





# In[ ]:





# In[ ]:




