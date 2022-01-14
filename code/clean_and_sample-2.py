#!/usr/bin/env python
# coding: utf-8

# Author: Thorsten Picard

# # Cleaning and sampling

# In this notebook we import the monthly datasets and perform a few operations on them that mainly involve removing outliers. After that, a sample is formed from smaller, random samples that are taken from the monthly datasets.
# 
# The preliminary cleaning operations on each monthly dataset involve the following:
# * Discard roundtrips (where station ID == end station ID);
# * Discard rows with a birth year that results in an age of 100 or larger;
# * Discard duplicate rows;
# * Discard rows containing NA values;
# * Discard rows where gender is not 1 (male) or 2 (female).

# ## Import packages

# In[6]:


# import packages
import pandas as pd
import numpy as np
from os import listdir
import time


# ## Clean and filter data

# In[2]:


# create a function that cleans the data

def clean_data(directory, file):
    
    # import data
    df = pd.read_csv(directory+file, header=0)
    
    # temporary variables
    df['roundtrip'] = df['start station id'] == df['end station id']
    df['age'] = (2019 - df['birth year']) >= 100
    
    # Get lengths
    len_df   = len(df) # row count original dataset
    len_NaN  = np.sum(df.isna().any(axis=1)) # NaNs
    len_dupl = np.sum(df.duplicated()) # duplicates
    len_M    = np.sum(df['gender']==1) # count males
    len_F    = np.sum(df['gender']==2) # count females
    len_O    = len_df - len_M - len_F # count other gender values
    len_RT   = np.sum(df['roundtrip']) # count roundtrips
    len_age_100  = np.sum(df['age']) # count age >= 100
    
    # drop rows with NA, drop duplicate rows
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # only include rows with gender == 1 (male) or 2 (female)
    df = df[df['gender'].isin([1, 2])]
    
    # discard roundtrips, discard rows with higher than boundary
    df = df[df['roundtrip'] == False]
    df = df[df['age'] == False]
    
    # get length of filtered dataset
    len_filt   = len(df) # row count filtered dataset
    len_filt_M = np.sum(df['gender']==1) # count males
    len_filt_F = np.sum(df['gender']==2) # count females
    
    # drop temporary columns
    cols_to_drop = ['roundtrip', 'age']
    df.drop(columns=cols_to_drop, inplace=True)
    
    name = int(file[4:6])
    
    # collect all counts 
    all_lengths = [name, 
                   len_df, len_NaN, len_dupl, len_M, len_F, len_O, len_RT, len_age_100, 
                   len_filt, len_filt_M, len_filt_F]
    
    return df, all_lengths


# In[3]:


# clean each monthly dataset

# settings
imp_dir = '../Data/'
exp_dir = '../data-2019-filtered/'
files_list = [f for f in listdir(imp_dir) if f.startswith('2019')]
files_list.sort()

# statistics about the number of discarded rows per month
d = []

# start measuring time
startTime = time.time()

print("PROGRESS")
print("-----------")

# conversion
for file in files_list:
    
    name = file[:6]

    # clean data and collect statistics about the number of discarded rows
    df, all_lengths = clean_data(imp_dir, file)
    
    # save cleaned data
    df.to_csv(exp_dir+str(name)+'.csv', header=True, index=False)
    del df
    
    # store statistics about the number of discarded rows
    d.append(all_lengths)
    
    print(name, "- done")

# stop measuring time and report
executionTime = time.time() - startTime
eT_min = np.round(executionTime / 60, 1)

print("-----------")
print("ALL DONE:", len(files_list), "files have been exported. Execution time:", np.round(executionTime, 0), 
      "seconds ("+str(eT_min)+" min).")


# statistics about the number of discarded rows for each month
cols = ['month', 
        'original_tripc', 'NaNs', 'duplicates', 'original_M', 'original_F', 'original_O', 
        'roundtrips','age_from_100', 
        'cleaned_tripc', 'cleaned_M', 'cleaned_F']

monthly_2019 = pd.DataFrame(np.array(d), columns=cols, dtype="int")
monthly_2019.to_csv('../summaries/submission_2/monthly_stats_cleaned.csv', header=True, index=False)


# ## Sample

# Let's now take a random sample from each monthly dataset. Then, combine all of them to have one, large sample.

# In[8]:


# random state
np.random.seed(6)
random_states = np.random.randint(low = 0, high = 100, size = 1)
print(random_states)


# In[3]:


# define directories
imp_dir = '../data-2019-filtered/'
exp_dir = '../samples/original/'
files_list = listdir(imp_dir)
files_list.sort()


# In[4]:


MONTH_SIZE = 50000

startTime = time.time()

for RS in random_states:
    
    sample_total = pd.DataFrame()
    
    for file in files_list:
        
        # import monthly dataset
        path = imp_dir+file
        df = pd.read_csv(path, header=0)
        
        # take monthly sample and add to total sample
        sample_month = df.sample(n=MONTH_SIZE, random_state=RS)
        sample_total = pd.concat([sample_total, sample_month])
    
    # set gender correctly
    sample_total['gender'] = sample_total['gender'] - 1
    
    # save total sample
    path = exp_dir + 'sample-rs-' + str(RS) + ".csv"
    sample_total.to_csv(path, header=True, index=False)
    
    print("Sample with random state", RS, "- saved.")   
    
executionTime = time.time() - startTime
eT_min = np.round(executionTime / 60, 2)

print("\nExecution time:", np.round(executionTime,0), 'sec (' + str(eT_min) + ' min).')


# Inspect the obtained sample.

# In[5]:


sample_total.head()


# In[ ]:





# In[ ]:




