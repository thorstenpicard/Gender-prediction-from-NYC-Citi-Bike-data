#!/usr/bin/env python
# coding: utf-8

# # 2019 Sampling

# In[99]:


# Import packages
import pandas as pd
import numpy as np
from os import listdir


# In[81]:


def sampling(directory, sample_size, random_state):
    
    total_sample = pd.DataFrame()
    
    files_list = listdir(directory)
    files_list.sort()
    file_count = len(files_list)
    
    print("In progress ...")
    print("------")

    for i in range(file_count):

        file = files_list[i]
        name = file[:6]

        df = pd.read_parquet(directory+file)
        
        if sample_size > len(df):
            month_sample = df.sample(n=len(df), random_state=random_state)
        else:
            month_sample = df.sample(n=sample_size, random_state=random_state)
        
        total_sample = pd.concat([total_sample, month_sample], ignore_index=True)
        
        del df, month_sample
        
        print(str(i) + '/' + str(file_count), name+' - done')

    print("------")
    print("Done!")
    
    return total_sample


# In[93]:


# Settings
directory = '../Data_2019_Converted/'
sample_size = 50000
random_state = 0

# Sample data
sample_2019 = sampling(directory, sample_size, random_state)
sample_2019.to_parquet('../Data_2019_Sample/CB-2019-sample.parquet', compression=None)


# In[94]:


sample_2019.shape


# In[95]:


sample_2019.info()


# In[96]:


sample_2019.head()


# In[97]:


sample_2019.tail()


# In[98]:


# Check share of female cyclists
neg, pos = np.bincount(sample_2019['gender'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive (female): {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


# In[ ]:




