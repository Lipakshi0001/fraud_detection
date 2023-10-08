#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[19]:


credit_card_data = pd.read_csv(r'C:\Users\admin\AppData\Local\Temp\Temp595ef9a3-2918-464c-ae96-9149760b54a0_creditcard.csv.zip\creditcard.csv')


# In[20]:


credit_card_data.head()


# In[21]:


credit_card_data.info()


# In[22]:


credit_card_data.isnull().sum()


# In[23]:


credit_card_data['Class'].value_counts()


# In[24]:


legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[25]:


print(legit.shape)
print(fraud.shape)


# In[26]:


legit.Amount.describe()


# In[27]:


fraud.Amount.describe()


# In[28]:


credit_card_data.groupby('Class').mean()


# In[29]:


legit_sample = legit.sample(n=492)


# In[30]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[31]:


new_dataset.head()


# In[32]:


new_dataset.tail()


# In[35]:


new_dataset['Class'].value_counts()


# In[37]:


new_dataset.groupby('Class').mean()


# In[38]:


x = new_dataset.drop(columns = 'Class', axis=1)
y = new_dataset['Class']


# In[39]:


print(x)


# In[40]:


print(y)


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2)


# In[44]:


print(x.shape, x_train.shape, x_test.shape)


# In[45]:


model = LogisticRegression()


# In[46]:


model.fit(x_train, y_train)


# In[47]:


x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)


# In[48]:


print('Accuracy on Training data = ', training_data_accuracy)


# In[49]:


x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)


# In[51]:


print('Accuracy on Testing data = ', test_data_accuracy)


# In[ ]:




