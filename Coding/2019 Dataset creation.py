#!/usr/bin/env python
# coding: utf-8

# # Creating the 2019 Citi Bike dataset

# ## Import packages and data

# In[26]:


# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn import preprocessing
from sklearn.metrics.pairwise import haversine_distances
from matplotlib.ticker import FuncFormatter
from math import radians

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Functions

# ### Filtering

# In[27]:


def exclude_data(directory, file):
    
    # Import data
    df = pd.read_csv(directory+file, header=0)
    
    # Temporary variables
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
    
    # Apply filters
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['gender'].isin([1, 2])]
    df = df[df['roundtrip'] == False]
    df = df[df['age'] == False]
    
    # Get length of filtered dataset
    len_filt   = len(df) # row count filtered dataset
    len_filt_M = np.sum(df['gender']==1) # count males
    len_filt_F = np.sum(df['gender']==2) # count females
    
    # Drop unnecessary columns
    cols_to_drop = ['roundtrip', 'bikeid', 'age']
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Sort + fix index
    df = df.sort_values(by=['starttime'])
    df.reset_index(drop=True, inplace=True)
    
    name = int(file[4:6])
    
    # Collect all counts 
    all_lengths = [name, 
                   len_df, len_NaN, len_dupl, len_M, len_F, len_O, len_RT, len_age_100, 
                   len_filt, len_filt_M, len_filt_F]
                   
    del name
    
    return df, all_lengths


# In[8]:


df, all_lengths = exclude_data(directory, file)
df.info()


# ### Distance (haversine)

# In[28]:


def distances(df):

    # Column indexes of start/and latitude/longitude
    ssla = df.columns.get_loc("start station latitude")
    sslo = df.columns.get_loc("start station longitude")
    esla = df.columns.get_loc("end station latitude")
    eslo = df.columns.get_loc("end station longitude")


    # Create two lists with start and end coordinates 
    starts = []
    ends = []

    for row in df.itertuples(index=False):
        start = list((row[ssla], row[sslo])) # start latitude, longitude
        starts.append(start)

        end = list((row[esla], row[eslo])) # end latitude, longitude
        ends.append(end)
        
        
    # Compute haversine distance    
    distances = []

    for i in range(len(df)):
        start_in_radians = [radians(_) for _ in starts[i]]
        end_in_radians   = [radians(_) for _ in ends[i]]

        result = haversine_distances([start_in_radians, end_in_radians])
        result = result * 6371000/1000 # multiply by Earth radius to get kilometers

        distances.append(result[0][1])
    
    df['distance_km'] = distances
    
    # Create shorter column names
    df.rename(columns = {'start station latitude': 'start_stat_lat', 
                         'start station longitude': 'start_stat_lon', 
                         'start station name': 'start_stat_name',
                         'start station id': 'start_stat_id',
                         'end station latitude': 'end_stat_lat', 
                         'end station longitude': 'end_stat_lon', 
                         'end station name': 'end_stat_name',
                         'end station id': 'end_stat_id', 
                         'birth year': 'birthyear'}, inplace = True)
    
    return df


# In[13]:


df2 = distances(df.copy())
df2.info()


# ### Time features

# In[29]:


def time_features(df):
    
    # Convert to timestamps
    df['starttime'] = pd.to_datetime(df['starttime'])
    df['stoptime']  = pd.to_datetime(df['stoptime'])

    # Get dates
    df['start_date'] = pd.to_datetime(df['starttime'].dt.date)
    df['stop_date']  = pd.to_datetime(df['stoptime'].dt.date)
    
    # Start and stop features
    for i in ('start', 'stop'):
        
        df[i+'_dayofyear']    = df[i+'time'].dt.dayofyear
        df[i+'_quarter']      = df[i+'time'].dt.quarter
        df[i+'_month']        = df[i+'time'].dt.month
        df[i+'_week']         = df[i+'time'].dt.isocalendar().week
        df[i+'_dayofmonth']   = df[i+'time'].dt.day
        df[i+'_weekday']      = df[i+'time'].dt.weekday # 0 = Monday, 6 = Sunday
        df[i+'_hour']         = df[i+'time'].dt.hour
        df[i+'_minute']       = df[i+'time'].dt.minute
        df[i+'_weekend']      = [1 if d >= 5 else 0 for d in df['start_weekday']]
    
    # Other time-related features
    df['trip_minutes'] = df['tripduration'] // 60
    df['speed_kmh'] = (df['distance_km'] / df['tripduration']) * 60 * 60
    
    # Drop original tripduration column
    df.drop(['tripduration'], axis=1, inplace=True)

    return df


# In[117]:


df3 = time_features(df2.copy())
df3.info()


# ## File conversion

# In[31]:


# Lists to collect data about the filter process
directory = '../Data_2019/'
files_list = os.listdir(directory)
files_list.sort()

# Collect monthly stats for df
d = []

###################################
# ------- CONVERT ALL FILES -------
###################################

print("PROGRESS")
print("-----------")

for file in files_list:
    
    name = file[:6]

    # Function 1
    df, all_lengths = exclude_data(directory, file) # Get df lengths and filtered df
    d.append(all_lengths)
    
    # Functions 2 and 3
    df = distances(df) # Compute haversine distance for each trip
    df = time_features(df) # Modify time features
    
    # Encode usertype and gender, move gender to end of df
    cleanup_ut = {"usertype": {"Subscriber": 0, "Customer": 1}}
    df = df.replace(cleanup_ut)
    
    le = preprocessing.LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    gender = df['gender']
    df.drop(['gender'], axis=1, inplace=True)
    df['gender'] = gender
    del gender
    
    # Convert dtypes
    cols = ['start_stat_name', 'end_stat_name']
    df[cols] = df[cols].astype('category')
    
    cols = ['start_stat_id', 'end_stat_id']
    df[cols] = df[cols].astype('int')
    del cols
    
    # Sort + fix index
    df = df.sort_values(by=['starttime'])
    df.reset_index(drop=True, inplace=True)
    
    # Save to parquet to maintain dtypes
    df.to_parquet('../Data_2019_Converted/'+str(name)+'.parquet', compression=None)
    
    del df
    
    print(name, "- done")

print("-----------")
print("ALL DONE:", len(files_list), "files have been exported to the folder 'Data_2019_Converted'")


###################################
# ------- DF MONTHLY STATS  -------
###################################

# Create dataframe from monthly stats
cols = ['month', 
        'original_tripc', 'NaNs', 'duplicates', 'original_M', 'original_F', 'original_O', 
        'roundtrips','age_from_100', 
        'filtered_tripc', 'filtered_M', 'filtered_F']

monthly_2019 = pd.DataFrame(np.array(d), columns=cols, dtype="int")
monthly_2019 = monthly_2019.sort_values(by=['month'])
monthly_2019.reset_index(drop=True, inplace=True)
monthly_2019.to_csv('../Summaries/cb-2019-monthly-results.csv', header=True, index=False)


# ## File size reduction + plotting

# ### Monthly stats

# In[61]:


monthly_2019 = pd.DataFrame(np.array(d), columns=cols, dtype="int")
monthly_2019 = monthly_2019.sort_values(by=['month'])
monthly_2019.reset_index(drop=True, inplace=True)
monthly_2019.to_csv('../Summaries/CB-2019-monthly-results.csv', header=True, index=False)
monthly_2019


# In[ ]:





# In[ ]:





# In[ ]:





# ### Plotting

# In[62]:


# Initiate plot
sns.set_theme()
fig, ax = plt.subplots()
sns.scatterplot(data=monthly_2019, x="month", y="original_tripc")

# Set thousand separators
ax.get_yaxis().set_major_formatter(
    FuncFormatter(lambda x, p: format(int(x), ',')))

plt.title("Monthly Citi Bike trip count in 2019", fontsize=16)
plt.xlabel("Month", fontsize = 14)
plt.ylabel("Trip count", fontsize = 14);


# ## Old code

# ### Distance (haversine)

# **Old distance formula**
# 
# def distances(df):
#     
#     df['distance_id'] = df.groupby(['startstationname', 'endstationname']).ngroup()
#     
#     df_locations = df[['startstationlatitude', 'startstationlongitude', 
#                        'endstationlatitude', 'endstationlongitude', 
#                        'distanceid']].drop_duplicates(subset=['distanceid'])
#     
#     # Create two lists: start and end coordinates
#     ssla = df_locations.columns.get_loc("startstationlatitude")
#     sslo = df_locations.columns.get_loc("startstationlongitude")
#     esla = df_locations.columns.get_loc("endstationlatitude")
#     eslo = df_locations.columns.get_loc("endstationlongitude")
# 
#     starts = [] 
#     ends = []
# 
#     for row in df_locations.itertuples(index=False):
#         start = list((row[ssla], row[sslo]))
#         starts.append(start)
# 
#         end = list((row[esla], row[eslo]))
#         ends.append(end)
#     
#     # Compute haversine distance    
#     distances = []
# 
#     for i in range(len(df_locations)):
#         start_in_radians = [radians(_) for _ in starts[i]]
#         end_in_radians   = [radians(_) for _ in ends[i]]
# 
#         result = haversine_distances([start_in_radians, end_in_radians])
#         result = result * 6371000/1000 # multiply by Earth radius to get kilometers
# 
#         distances.append(result[0][1])
#         
#     df_locations['distance_km'] = distances
#     
#     df = df.join(df_locations.iloc[:,-2:].set_index('distance_id'), on='distance_id')
#     del df_locations
#     
#     # Drop columns
#     cols_to_drop = ['startstationlatitude', 'startstationlongitude', 
#                     'endstationlatitude', 'endstationlongitude', 'distance_id']
# 
#     return df.drop(cols_to_drop, axis=1, inplace=True)

# ### Cyclical nature of time

# **Cyclical nature of time**
# 
#     # Capture cyclical nature of time
#     df['start_hr_sin'] = np.sin(df.start_hr.astype(int)*(2.*np.pi/24))
#     df['start_hr_cos'] = np.cos(df.start_hr.astype(int)*(2.*np.pi/24))
#     year = df['starttime'].dt.year[0]
#     month_size = monthrange(year, df['mnth'][0])[1] # number of days in month
#     df['start_day_sin'] = np.sin(df.start_day.astype(int)*(2.*np.pi/month_size))
#     df['start_day_cos'] = np.cos(df.start_day.astype(int)*(2.*np.pi/month_size))
#     df['weekday_sin'] = np.sin(df.weekday.astype(int)*(2.*np.pi/7))
#     df['weekday_cos'] = np.cos(df.weekday.astype(int)*(2.*np.pi/7))
#     df['mnth_sin'] = np.sin((df.mnth.astype(int)-1)*(2.*np.pi/12))
#     df['mnth_cos'] = np.cos((df.mnth.astype(int)-1)*(2.*np.pi/12))  
#     
#     return df

# ### Conversion function

# def convert(file):
#     
#     name = file[:6]
#     
#     # Import data and make columns consistent
#     df = pd.read_csv(directory+file, header=0)
#     df.columns = df.columns.str.replace(" ", "")
#     df.columns = df.columns.str.strip().str.lower()
#             
#     # Drop columns
#     cols_to_drop = ['startstationid', 'endstationid', 'bikeid']
#     df.drop(cols_to_drop, axis=1, inplace=True)
# 
#     # Filter
#     df.dropna(inplace=True)
#     gender_list = [1, 2] # male = 1, female = 2
#     df = df[(df['usertype']=='Subscriber') & (df['gender'].isin(gender_list))]
#     
#     # Drop columns
#     cols_to_drop = ['usertype']
#     df.drop(cols_to_drop, axis=1, inplace=True)
#     df.reset_index(drop=True, inplace=True)
#             
#     # Convert time stamps
#     df['starttime'] = pd.to_datetime(df['starttime'])
#     df['stoptime']  = pd.to_datetime(df['stoptime'])
# 
#     # Encode gender
#     le = LabelEncoder()
#     df['gender'] = le.fit_transform(df['gender'])
# 
#     # Categorise variables
#     cols = ['startstationname', 'endstationname']
#     for col in cols:
#         df[col] = df[col].astype('category')
#     
#     
#     # DISTANCE
#     ################################################
# 
#     df['distanceid'] = df.groupby(['startstationname', 'endstationname']).ngroup()
#     df_locations = df[['startstationlatitude', 'startstationlongitude', 
#                        'endstationlatitude', 'endstationlongitude', 
#                        'distanceid']].drop_duplicates(subset=['distanceid'])
#     
#     # Create two lists: start and end coordinates
#     ssla = df_locations.columns.get_loc("startstationlatitude")
#     sslo = df_locations.columns.get_loc("startstationlongitude")
#     esla = df_locations.columns.get_loc("endstationlatitude")
#     eslo = df_locations.columns.get_loc("endstationlongitude")
# 
#     starts = [] 
#     ends = []
# 
#     for row in df_locations.itertuples(index=False):
#         start = list((row[ssla], row[sslo]))
#         starts.append(start)
# 
#         end = list((row[esla], row[eslo]))
#         ends.append(end)
#     
#     # Compute haversine distance    
#     distances = []
# 
#     for i in range(len(df_locations)):
#         start_in_radians = [radians(_) for _ in starts[i]]
#         end_in_radians   = [radians(_) for _ in ends[i]]
# 
#         result = haversine_distances([start_in_radians, end_in_radians])
#         result = result * 6371000/1000 # multiply by Earth radius to get kilometers
# 
#         distances.append(result[0][1])
#         
#     df_locations['distancekm'] = distances
#     
#     df = df.join(df_locations.iloc[:,-2:].set_index('distanceid'), on='distanceid')
#     del df_locations
#     
#     # Drop columns
#     cols_to_drop = ['startstationlatitude', 'startstationlongitude', 
#                     'endstationlatitude', 'endstationlongitude', 'distanceid']
# 
#     df.drop(cols_to_drop, axis=1, inplace=True)
#     
#     
#     # TIME
#     ################################################
#     
#     # New features
#     df['trip_min']   = df['tripduration'] // 60
#     df['start_hr']   = df['starttime'].dt.hour.astype('category')
#     df['start_day']  = df['starttime'].dt.day.astype('category')
#     df['weekday']    = df['starttime'].dt.weekday.astype('category')
#     df['mnth']       = df['starttime'].dt.month.astype('category')
#     df['wknd']       = [1 if d >= 5 else 0 for d in df['weekday']]
#     df['speedkmh'] = (df['distancekm'] / df['tripduration']) * 60 * 60
# 
#     df.drop(['tripduration'], axis=1, inplace=True)
#     
#     # Capture cyclical nature of time
#     df['start_hr_sin'] = np.sin(df.start_hr.astype(int)*(2.*np.pi/24))
#     df['start_hr_cos'] = np.cos(df.start_hr.astype(int)*(2.*np.pi/24))
#     year = df['starttime'].dt.year[0]
#     month_size = monthrange(year, df['mnth'][0])[1] # number of days in month
#     df['start_day_sin'] = np.sin(df.start_day.astype(int)*(2.*np.pi/month_size))
#     df['start_day_cos'] = np.cos(df.start_day.astype(int)*(2.*np.pi/month_size))
#     df['weekday_sin'] = np.sin(df.weekday.astype(int)*(2.*np.pi/7))
#     df['weekday_cos'] = np.cos(df.weekday.astype(int)*(2.*np.pi/7))
#     df['mnth_sin'] = np.sin((df.mnth.astype(int)-1)*(2.*np.pi/12))
#     df['mnth_cos'] = np.cos((df.mnth.astype(int)-1)*(2.*np.pi/12))
#     
#     
#     # FINALISE
#     ################################################
#     
#     # Move gender to end of df
#     gender = df['gender']
#     df.drop(['gender'], axis=1, inplace=True)
#     df['gender'] = gender
#     del gender
# 
#     # Sort + fix index
#     df = df.sort_values(by=['starttime'])
#     df.reset_index(drop=True, inplace=True)
#     
#     # Save to parquet to maintain dtypes
#     df.to_parquet('./Data_Converted/'+str(name)+'.parquet', compression=None)
#     del df 

# In[ ]:





# In[ ]:




