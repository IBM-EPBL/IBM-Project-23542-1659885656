#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries


# In[2]:


import pandas as pd
import numpy as np
from collections import Counter as c
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle


# In[3]:


# Loading the dataset


# In[10]:


data=pd.read_csv(r"C:\Users\Datasets\chronickidneydisease.csv")


# In[11]:


data.head() # return you the first 5 rows values


# In[12]:


data.tail() # return you the last 5 rows values


# In[13]:


data.head(10) # return the first 10 rows values


# In[14]:


data.drop(["id"],axis=1,inplace=True) # drop is used for drop the column


# In[15]:


data.columns # return all the column names


# In[16]:


data.columns=['age','blood_pressure','specific_gravity','albumin','sugar','red_blood_cells','pus_cell','pus_cell_clumps','bacteria','blood glucose random',
'blood_urea','serum_creatinine','sodium','potassium','hemoglobin','packed_cell_volume','white_blood_cell_count','red_blood_cell_count','hypertension',
'diabetesmellitus','coronary_artery_disease','appetite','pedal_edema','anemia','class'] # manually giving the name of the columns
data.columns


# In[17]:


data.info() # info will give you a summary of dataset


# In[18]:


# Target Column


# In[19]:


data['class'].unique() # find the unique elements of an array


# In[20]:


data['class']=data['class'].replace("ckd\t","ckd") # replace is used for renaming
data['class'].unique()


# In[21]:


catcols = set(data.dtypes[data.dtypes =='O'].index.values) # only fetch the object type columns
print(catcols)


# In[22]:


for i in catcols:
 print("Columns :",i)
 print(c(data[i])) # using counter for checking the number of classess in the column
 print('*'*120+'\n')


# In[24]:


# Understanding Datatype and summary of features


# In[25]:


# Removing the Columns which are not Numerical
# Categorical Column


# In[26]:


catcols.remove('red_blood_cell_count')
catcols.remove('packed_cell_volume')
catcols.remove('white_blood_cell_count')
print(catcols)


# In[27]:


# Numerical Column


# In[28]:


contcols=set(data.dtypes[data.dtypes!='O'].index.values) # only fetch the float and int type columns
#contcols=pd.DataFrame(data,columns=contcols)
print(contcols)


# In[29]:


contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')
print(contcols)


# In[30]:


# Adding columns which we found continuous


# In[31]:


contcols.add('red_blood_cell_count') # using add we can add the column
contcols.add('packed_cell_volume')
contcols.add('white_blood_cell_count')
print(contcols)


# In[32]:


# Adding columns which we found Categorical


# In[33]:


catcols.add('specific_gravity')
catcols.add('albumin')
catcols.add('sugar')
print(catcols)


# In[34]:


# Rectifying the Categorical column classes


# In[35]:


data['coronary_artery_disease'] = data.coronary_artery_disease.replace('\tno','no') # replacing \tno with no
c(data['coronary_artery_disease'])


# In[36]:


data['diabetesmellitus'] = data.diabetesmellitus.replace(to_replace={'\tno':'no','\tyes':'yes','yes':'yes'})
c(data['diabetesmellitus'])


# In[37]:


# Handling the missing values


# In[38]:


## Null Values


# In[39]:


data.isnull().any() # it will return true if any missing values


# In[40]:


data.isnull().sum() # returns the count of missing values


# In[41]:


data.packed_cell_volume = pd.to_numeric(data.packed_cell_volume,errors='coerce')
data.white_blood_cell_count = pd.to_numeric(data.white_blood_cell_count,errors='coerce')
data.red_blood_cell_count = pd.to_numeric(data.red_blood_cell_count,errors='coerce')


# In[42]:


# Replacing the missing values


# In[43]:


# Handling Continuous/numerical colunmns null values


# In[44]:


data['blood glucose random'].fillna(data['blood glucose random'].mean(),inplace=True)
data['blood_pressure'].fillna(data['blood_pressure'].mean(),inplace=True)
data['blood_urea'].fillna(data['blood_urea'].mean(),inplace=True)
data['hemoglobin'].fillna(data['hemoglobin'].mean(),inplace=True)
data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(),inplace=True)
data['potassium'].fillna(data['potassium'].mean(),inplace=True)
data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(),inplace=True)
data['serum_creatinine'].fillna(data['serum_creatinine'].mean(),inplace=True)
data['sodium'].fillna(data['sodium'].mean(),inplace=True)
data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(),inplace=True)


# In[45]:


data['age'].fillna(data['age'].mode()[0],inplace=True)
data['hypertension'].fillna(data['hypertension'].mode()[0],inplace=True)
data['pus_cell_clumps'].fillna(data['pus_cell_clumps'].mode()[0],inplace=True)
data['appetite'].fillna(data['appetite'].mode()[0],inplace=True)
data['albumin'].fillna(data['albumin'].mode()[0],inplace=True)
data['pus_cell'].fillna(data['pus_cell'].mode()[0],inplace=True)
data['red_blood_cells'].fillna(data['red_blood_cells'].mode()[0],inplace=True)
data['coronary_artery_disease'].fillna(data['coronary_artery_disease'].mode()[0],inplace=True)
data['bacteria'].fillna(data['bacteria'].mode()[0],inplace=True)
data['anemia'].fillna(data['anemia'].mode()[0],inplace=True)
data['sugar'].fillna(data['sugar'].mode()[0],inplace=True)
data['diabetesmellitus'].fillna(data['diabetesmellitus'].mode()[0],inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0],inplace=True)
data['specific_gravity'].fillna(data['specific_gravity'].mode()[0],inplace=True)


# In[46]:


data.isnull().sum()


# In[47]:


# Label Encoding


# In[48]:


from sklearn.preprocessing import LabelEncoder # importing Labelencoding from sklearn
for i in catcols: # looping through all the categorical column
 print("LABEL ENCODING OF:",i)
LEi = LabelEncoder() # creating an object of labelencoder
print(c(data[i])) # getting the classes values before transformation 
data[i] = LEi.fit_transform(data[i]) # transfering our test classes to numerical values
print(c(data[i])) # getting the classes values after transformation
print("*"*100)


# In[49]:


# Splitting the dataset into dependent and independant variable


# In[50]:


# Creating Independent and Dependent


# In[51]:


selcols=['red_blood_cells','pus_cell','blood glucose random','blood_urea','pedal_edema','anemia','diabetesmellitus','coronary_artery_disease']
x=pd.DataFrame(data,columns=selcols)
y=pd.DataFrame(data,columns=['class'])
print(x.shape)
print(y.shape)


# In[52]:


# Splitting the Data into train and test


# In[53]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2) # train test split the data


# In[54]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:




