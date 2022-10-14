#!/usr/bin/env python
# coding: utf-8

# ### Data Preprocessing ###

# #### Data Cleaning ####

# * Noisy Data
# * Missing Data Analysis
# * Outlier Analysis

# #### Data Standardization - Feature Scaling ####

# * Normalizasyon(0-1)
# * Standardization(z-score)
# * Log Transformation

# #### Data Reduction ####

# * Reducing the number of observations
# * Reducing the number of variables

# #### Variable Transformation ####

# * Transformation for Categorical Variablesm

# ### For single variable ###

# #### Outlier variables/observations ####

# Threshold Value

# We define threshold value,the mean of numbers and add one,two or three std.In addition we define outlier value the high and low value of this value.

# Threshold Value = Mean + 1 X Std,
# Threshold Value = Mean + 2 X Std or
# Threshold Value = Mean + 3 X Std

# #### Standardization(z-score) ####

# Value implement normal distribution so İt is standardized and we define threshold value adding +-2 from this distribution.In addition we define outlier value the high and low value of this value.m 

# #### Boxplot(Interquartile Range-IQR) ####

# In[32]:


import seaborn as sns
import pandas as pd
import numpy as np


# In[33]:


tip=sns.load_dataset('tips')
tip


# In[54]:


tip_sex=tip['size']


# In[55]:


sns.boxplot(x=tip_size);


# * IQR=1,5X(Q3-Q1)
# * Lower threshold value=Q1-IQR
# * Upper threshold value=Q3+IQR

# In[35]:


tip


# 'select_dtypes' select that you select data types

# In[42]:


tip=tip.select_dtypes(include = ['float64', 'int64'])
tip


# 'dropna()' delete missing value in a data

# In[45]:


tip=tip.dropna()
tip


# In[46]:


tip.head(10)


# In[52]:


tip_size=tip['size']
tip_size.head(10)


# In[57]:


sns.boxplot(x=tip_size);


# In[59]:


Q1=tip_size.quantile(0.25)
Q1


# In[61]:


Q3=tip_size.quantile(0.75)
Q3


# In[64]:


IQR=(Q3-Q1)*1.5
IQR


# * Lower threshold value=Q1-IQR
# * Upper threshold value=Q3+IQR

# In[65]:


lo=Q1-IQR
lo


# In[69]:


up=Q3+IQR
up


# In[72]:


di=tip_size<lo
di


# list of lower threshold value

# In[74]:


tip_size[di]


# In[75]:


di_2=tip_size>up
di_2


# In[76]:


tip_size[di_2]


# In[77]:


tip_size[di_2].index


# #### 1- remove ####

# The simplest solution is to remove that observation.

# In[78]:


tip_size.shape


# In[79]:


tip_size=pd.DataFrame(tip_size)


# In[80]:


tip_size.shape


# In[82]:


tip_size_t=tip_size[~((tip_size<lo)|(tip_size>up)).any(axis=1)]
tip_size_t


# '~' selects values within threshold values.

# In[83]:


tip_size_t.shape


# In[85]:


tip_size.shape


# #### 2-Fill the mean ####

# Another solution is to use a global constant to fill that gap, like “NA” or 0, An alternative option is to use the mean or median of that attribute to fill the gap. 

# In[87]:


tip_size[di]


# In[ ]:


# tip_size[di_2]


# In[95]:


# tip_size[di_2]=tip_size.mean()


# #### Suppression of Data ####

# Using the backward/forward fill method is another approach that can be applied.

# In[98]:


tip_2=sns.load_dataset('tips')
tip_2


# In[101]:


tip_2=tip_2.select_dtypes(include=['float64', 'int64'])
tip_2


# In[103]:


tip_2=tip_2.dropna()
tip_2


# In[104]:


tip_2.head(10)


# In[107]:


tip_2_size=tip_2['size']
tip_2_size


# In[108]:


Q1=tip_2_size.quantile(0.25)
Q1


# In[109]:


Q2=tip_2_size.quantile(0.25)
Q2


# In[110]:


IQR=(Q3-Q1)*1.5
IQR


# In[111]:


lo_2=Q1-IQR
lo_2


# In[112]:


up_2=Q3+IQR
up_2


# In[114]:


di_2=tip_2_size<lo_2
di_2


# In[115]:


tip_size[di_2]


# In[116]:


di_3=tip_2_size>up_2
di_3


# In[117]:


tip_2_size[di_3]


# In[118]:


tip_2_size[di_3]=up_2


# In[119]:


tip_2_size[di_3]


# ### For multiple variables ###

# #### Local Outlier Factor ####

# In[120]:


from sklearn.neighbors import LocalOutlierFactor


# In[121]:


clf=LocalOutlierFactor(n_neighbors=20, contamination=0.1)


# In[124]:


clf.fit_predict(tip)


# In[127]:


tip_score=clf.negative_outlier_factor_
tip_score[:10]


# In[129]:


np.sort(tip_score)[:10]


# In[131]:


threshold_value=np.sort(tip_score)[1]
threshold_value


# In[137]:


di_4=tip_score<threshold_value
di_4


# In[163]:


a=tip.to_records(index=False)
a


# In[164]:


a[di_4]


# In[165]:


a[di_5]


# In[168]:


a[di_4]=a[di_5]


# In[169]:


a[di_4]==a[di_5]


# In[171]:


a[di_4]


# In[172]:


a[di_5]


# In[174]:


tips=pd.DataFrame(a)
tips

