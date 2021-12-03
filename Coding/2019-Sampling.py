#!/usr/bin/env python
# coding: utf-8

# # 2019 Sampling

# In[1]:


# Import packages
import pandas as pd
import numpy as np
from os import listdir


# In[2]:


def sampling(directory, sample_size, random_state):
    
    """ FUNCTION DESCRIPTION
    • Purpose: create a sample from the 2019 parquet files
    • Input: directory name, sample size, random state
    • Output: dataframe with a sample of 2019 data
    • Process: import each file, take sample_size, include in dataframe
    """
    
    # Initiate df to collect a sample of 2019 data
    total_sample = pd.DataFrame()
    
    # Get parquet files in directory
    files_list = listdir(directory)
    files_list.sort() # sort now to avoid sorting the total_sample on date
    file_count = len(files_list)
    
    print("Sampling in progress ...")
    print("------")

    
    for i in range(file_count):

        """
        Import every file, exctract a random sample_size of instances from the data. 
        If sample size >= length of dataframe, extract entire dataframe.
        """
        
        file = files_list[i]
        name = file[:6]

        df = pd.read_parquet(directory+file)
        
        if sample_size >= len(df):
            month_sample = df.sample(n=len(df), random_state=random_state)
        else:
            month_sample = df.sample(n=sample_size, random_state=random_state)
        
        # add sub sample to total sample
        total_sample = pd.concat([total_sample, month_sample], ignore_index=True)
        
        # delete bc memory
        del df, month_sample
        
        # print that current file is done
        print(str(i) + '/' + str(file_count), name+' - done')

    print("------")
    print("Done!\n")
    
    return total_sample


# Settings
directory = '../Data_2019_Converted/'
sample_size = 50000 # total sample size = sample_size * count_files
random_state = 0 # change to get a different sample

# Get a sample from 2019
sample_2019 = sampling(directory, sample_size, random_state)
print("Shape of 2019 sample:", sample_2019.shape, '\n')
sample_2019.to_parquet('../Data_2019_Sample/cb-2019-sample-rs-0.parquet', compression=None)

# Check gender frequency
print("Please check if sample represents gender frequency of original dataset:")
print("  Statistics original 2019 dataset: 74.7\% male, 25.3\% female.\n")

neg, pos = np.bincount(sample_2019['gender'])
total = neg + pos
print("2019 sample:")
print('Examples:\n    Total: {}\n    Positive (female): {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


# In[3]:





# In[98]:





# In[ ]:




