#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Title: GENDER PREDICTION FROM NEW YORK CITY’S BIKE-SHARING DATA: 
#        A MACHINE LEARNING AND DEEP LEARNING APPROACH

# Author: Thorsten Picard
# Programme: MSc. Data Science and Society
# University: Tilburg University


# **READ ME**
# 
# This file imports the raw monthly Citi Bike datasets from 2019, performs a few basic cleaning operations, and takes a sample from the cleaned data. The cleaning operations are the following:
# * Discard roundtrips (where station ID == end station ID)
# * Discard rows with a birth year that results in an age of 100 or larger
# * Discard duplicate rows
# * Discard rows containing NA values
# * Discard rows where gender is not 1 (male) or 2 (female)
# 
# The changes these operations bring about are stored in a file called "monthly_stats_cleaned.csv". This file allows anyone to see by how many rows the monthly datasets have been reduced.

# In[14]:


# import packages
import pandas as pd
import numpy as np
from os import listdir
import time


# ## Clean and filter data

# In[15]:


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


# In[16]:


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

# Given the sheer volume of data (+ 18 million rows), samples will be subtracted from the 2019 dataset. A balanced and an unbalanced sample in terms of the target variable (gender) will be taken from the 2019 dataset for each random state.

# In[22]:


# obtain a random state, change size to increase the number of samples per sample type (bal, unb).  

np.random.seed(6)
random_states = np.random.randint(low = 0, high = 100, size = 1)

print(random_states)


# In[23]:


# ------- FILE SETTINGS -------
# files
imp_dir = '../data-2019-filtered/'
exp_dir = '../samples/original/'
files_list = listdir(imp_dir)
files_list.sort()


# ------- SAMPLING -------

# sample settings 
total_sample_size = 50000 # per month
test_sample_size = int(.3 * total_sample_size)
train_sample_size = int(total_sample_size - test_sample_size)

startTime = time.time()

print("PROGRESS")
print("-----------")

count = 0

for i in range(len(random_states)):
    
    rs = random_states[i]

    # create one sample with all monthly samples
    total_sample_unb = pd.DataFrame()
    total_sample_bal = pd.DataFrame()
    
    # iterate over the original, monthly datasets
    for file in files_list:

        # import monthly dataset
        path = imp_dir+file
        df = pd.read_csv(path, header=0)
        
        # ---- UNBALANCED
        sample_unb = df.sample(n=total_sample_size, random_state=rs)
        total_sample_unb = pd.concat([total_sample_unb, sample_unb])
        
        # ---- BALANCED
        # test set, unbalanced
        test_data = df.sample(n=test_sample_size, random_state=rs)
        test_data['set'] = ['test'] * test_data.shape[0]
        test_indices = test_data.index.values # get indices of test set
        df_new = df.drop(test_indices) # drop test set from original dataset
        
        # val set, unbalanced
        val_data  = df_new.sample(n=int(.3*train_sample_size), random_state=rs)
        val_data['set'] = ['val'] * val_data.shape[0]
        val_indices = val_data.index.values # get indices of test set
        df_new2 = df_new.drop(val_indices) # drop test set from original dataset        

        # training set, balanced
        male, female = 1, 2
        df_new_M = df_new2[df_new2['gender']==male] # male set
        df_new_F = df_new2[df_new2['gender']==female] # female set
            
        train_data = pd.DataFrame()
        for gender_set in (df_new_M, df_new_F):
            train_data = pd.concat([train_data, 
                                    gender_set.sample(n=int((.7*train_sample_size)/2), random_state=rs)])
        train_data['set'] = ['train'] * train_data.shape[0]
        
        # check for duplicates
        all_indices = np.concatenate((train_data.index, val_data.index, test_data.index), axis=None)
        count_duplicates = np.sum(pd.Series(all_indices).duplicated())
            
        if count_duplicates == 0:
            total_sample_bal = pd.concat([total_sample_bal, train_data, val_data, test_data])
        else:
            print("duplicates exist between train and test sets")

    # save total samples
    total_sample_bal.to_csv(exp_dir+"bal-sample-"+str(i)+".csv", header=True, index=False)
    count += 1
    total_sample_unb.to_csv(exp_dir+"unb-sample-"+str(i)+".csv", header=True, index=False)
    count += 1
    
    print("Sample with random state "+str(rs)+" - done.")

executionTime = time.time() - startTime
eT_min = np.round(executionTime / 60, 2)

print("-----------")
print("ALL DONE - "+ str(count) +" samples have been exported. Execution time:", np.round(executionTime, 0), 
      "seconds ("+str(eT_min)+" min).")

