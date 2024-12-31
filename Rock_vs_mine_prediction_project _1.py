#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# Loading the dataset to a pandas dataframe.
sonar_data = pd.read_csv('data(Rock_vs_mine_prediction_project_#1)/Sonar data.csv', header=None)


# In[3]:


# Prints first five rows.
sonar_data.head()


# In[4]:


# No. of rows and cols.
sonar_data.shape


# In[5]:


# describr --> statistical measures of the data.
sonar_data.describe()


# In[6]:


# Checking if its an imbalanced dataset.
sonar_data[60].value_counts()


# In[7]:


# Approximately the same no. of records for mines and rocks.
# M --> Mine
# R --> Rock


# In[8]:


sonar_data.groupby(60).mean()
# Mean of both rock and mines records in the 60th column.


# In[9]:


# Seperating data and labels.
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]


# In[10]:


print(X)
print(Y)


# ### Training and Test data

# In[11]:


# The parameter stratify divides the data such that the proportion of each class in the specified column is the same in both training and test datasets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)


# In[12]:


print(X.shape, X_train.shape, X_test.shape)


# ### Model training--> Logistic Regression

# In[13]:


model = LogisticRegression()


# In[14]:


# Training the model with training data.
model.fit(X_train, Y_train)


# ### Model Evaluation

# In[15]:


# Accuracy on training data.
X_train_predictions = model.predict(X_train)
X_train_predictions_accuracy = accuracy_score(X_train_predictions, Y_train)


# In[16]:


print('Accuracy on training data: ' , X_train_predictions_accuracy)


# In[17]:


# Accuracy is almost 80% on training data.


# In[18]:


# Accuracy on test data.
X_test_predictions = model.predict(X_test)
X_test_predictions_accuracy = accuracy_score(X_test_predictions, Y_test)


# In[19]:


print('Accuracy on test data: ', X_test_predictions_accuracy)


# In[20]:


# Accuracy on test data is approximately 76%


# ### Making a predictive system

# In[21]:


# Input data here for prediction:
input_data = ()

# Changing the input_data to a numpy array.
input_data_as_numpy_array = np.asarray(input_data)

# Rehaping the np array as we are predicting for one instance.
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

predection = model.predict(input_data_reshaped)
print(predection)

if (predection[0]=='R'):
    print('The object is a rock')
else:
    print('The object is a mine')

