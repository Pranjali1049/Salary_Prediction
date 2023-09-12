#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Importing Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error 


# In[37]:


# Importing Data
df = pd.read_csv(r'C:\Users\acer\Downloads\Salary_Data.csv')
df


# In[38]:


df.info()


# In[39]:


df.describe()


# In[40]:


df.isnull().sum()


# In[41]:


df.dropna(inplace=True)


# In[42]:


df['Job Title'].value_counts()


# In[43]:


# Reducing Job titles by omitting titles with less than 25 counts
job_title_count = df['Job Title'].value_counts()
job_title_edited = job_title_count[job_title_count<=25]
job_title_edited.count()


# In[44]:


# Omitting titles with less than 20 counts
df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited else x )
df['Job Title'].nunique()


# In[45]:


df['Education Level'].value_counts()


# In[46]:


# Combining repeating values of education level
df['Education Level'].replace(["Bachelor's Degree","Master's Degree","phD"],["Bachelor's","Master's","PhD"],inplace=True)
df['Education Level'].value_counts()


# In[47]:


df['Gender'].value_counts()


# In[ ]:




