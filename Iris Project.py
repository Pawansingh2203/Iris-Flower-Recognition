#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_iris


# In[4]:


iris_dataset=load_iris()
type(iris_dataset)


# In[5]:


print(iris_dataset.keys())


# In[6]:


print('Discription of dataset \n{}'.format(iris_dataset['DESCR'][:193]))


# In[7]:


print('Target Names\n{}'.format(iris_dataset['target_names']))


# In[8]:


print('fearture Names \n{}'.format(iris_dataset['feature_names']))


# In[9]:


print('Shape of Data \n{}'.format(iris_dataset['data'].shape))


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print('X Train set {}. y train set {}.'.format(X_train.shape,y_train.shape))
print('X Test set {}. y test set {}.'.format(X_test.shape,y_test.shape))


# In[11]:


import pandas as pd
import mglearn


# In[12]:


iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe,alpha=0.8,c=y_train,figsize=(10,10),marker='o',hist_kwds={'bins':20}, s=60,cmap=mglearn.cm3)


# In[13]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[14]:


x_new=np.array([[5,2.9,1,0.2]])
print('X_new shape: {} '.format(x_new.shape))


# In[15]:


prediction=knn.predict(x_new)
print('Prediction {}'.format(prediction))
print('Prediction name {}'.format(iris_dataset['target_names'][prediction]))


# In[18]:


y_predict=knn.predict(X_test)
print('Test Set Score {:.2f}'.format(np.mean(y_predict==y_test)))


# In[ ]:




