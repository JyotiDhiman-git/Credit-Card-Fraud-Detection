#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('creditcard.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


print("Number of Rows",df.shape[0])
print("Number of Columns",df.shape[1])


# In[7]:


df.info()


# In[8]:


df.describe()


# # Check Missing Values in data

# In[9]:


df.isnull().sum()


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


sc = StandardScaler()
df['Amount']=sc.fit_transform(pd.DataFrame(df['Amount']))


# In[12]:


df.sample(5)


# In[13]:


df = df.drop(['Time'],axis=1)


# In[14]:


df.head()


# In[15]:


df.duplicated().any()


# In[16]:


df.duplicated()


# In[17]:


df = df.drop_duplicates()


# In[18]:


df.shape


# In[19]:


284807- 275663


# # Not Handling Imbalanced

# In[20]:


df['Class'].value_counts()


# # Visualization

# In[22]:


sns.countplot(df['Class'])


# # Store Feature Matrix In X And Response (Target) In Vector y

# In[23]:


X = df.drop('Class',axis=1)
y = df['Class']


# # Splitting The Dataset Into The Training Set And Test Set

# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# # Handling Imbalanced Dataset
1.Undersampling
2.Oversampling
# ## Undersampling

# In[26]:


normal = df[df['Class']==0]
fraud = df[df['Class']==1]


# In[28]:


normal.shape


# In[29]:


fraud.shape


# In[30]:


normal_sample=normal.sample(n=473)


# In[31]:


normal_sample.shape


# In[32]:


new_df = pd.concat([normal_sample,fraud],ignore_index=True)


# In[33]:


new_df['Class'].value_counts()


# In[34]:


new_df.head()


# In[35]:


X = new_df.drop('Class',axis=1)
y = new_df['Class']


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# ##  Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score


# In[40]:


accuracy_score(y_test,y_pred1)


# In[42]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[43]:


precision_score(y_test,y_pred1)


# In[50]:


precision_score(y_test,y_pred1)


# In[51]:


recall_score(y_test,y_pred1)


# In[52]:


f1_score(y_test,y_pred1)


# In[53]:


f1_score(y_test,y_pred1)


# ## Decision Tree Classifier

# In[58]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[59]:


y_pred2 = dt.predict(X_test)


# In[60]:


accuracy_score(y_test,y_pred2)


# In[61]:


precision_score(y_test,y_pred2)


# In[62]:


recall_score(y_test,y_pred2)


# In[63]:


f1_score(y_test,y_pred2)


# ## Random Forest Classifier

# In[64]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[65]:


y_pred3 = rf.predict(X_test)


# In[66]:


accuracy_score(y_test,y_pred3)


# In[67]:


precision_score(y_test,y_pred3)


# In[68]:


recall_score(y_test,y_pred3)


# In[69]:


f1_score(y_test,y_pred3)


# In[70]:


final_data = pd.DataFrame({'Models':['LR','DT','RF'],"ACC":[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})
final_data


# In[71]:


sns.barplot(final_data['Models'],final_data['ACC'])


# In[ ]:




