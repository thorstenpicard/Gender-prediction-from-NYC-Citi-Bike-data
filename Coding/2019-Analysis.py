#!/usr/bin/env python
# coding: utf-8

# # Citi Bike 2019 Analysis

# ## Packages and data import

# In[228]:


# Import packages
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from matplotlib.ticker import FuncFormatter
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import skew, kurtosis
from os import listdir

from sklearn import metrics, tree
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from mixed_naive_bayes import MixedNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# MLP packages
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers
from tensorflow.keras import Sequential


# In[2]:


# import data
df_sample = pd.read_parquet('../Data_2019_Sample/cb-2019-sample-rs-0.parquet')
df_monthly = pd.read_csv('../Summaries/cb-2019-monthly-results.csv') # originally df_pr


# In[188]:


df_sample.shape


# ## Difference between original and filtered dataset
# This section quantifies the differences between the original and the filtered dataset based on the aggregated  statistics in df_monthly. 

# In[14]:


# GENDER RATIOS
print("GENDER RATIOS\n-----------")
or_ratio_MF = np.mean(df_monthly.original_M / df_monthly.original_F)
fi_ratio_MF = np.mean(df_monthly.filtered_M / df_monthly.filtered_F)

print("Original ratio M:F", str(np.round(or_ratio_MF, 3))+':1')
print("Filtered ratio M:F", str(np.round(fi_ratio_MF, 3))+':1', '\n')

TOT = np.sum(df_monthly.original_tripc)

# ROUND TRIPS
print("ROUNDTRIPS\n-----------")
RT = np.sum(df_monthly['roundtrips'])
print("Total trips:", f'{TOT:,d}')
print("Roundtrips:", f'{RT:,d}')
print("Count after RTs:", f'{TOT-RT:,d}')
print(str(np.round((RT/TOT) * 100,2))+'%\n')

# GENDER
print("GENDER\n-----------")
MF = np.sum(np.sum(df_monthly[['original_F', 'original_M']]))
not_MF = TOT - MF 
print("Total trips:", f'{TOT:,d}')
print("Not MF:", f'{not_MF:,d}')
print("Count after not_MF:", f'{TOT-not_MF:,d}')
print(str(np.round((not_MF/TOT) * 100,2))+'%\n')

# GENDER
print("BIRTH YEAR\n-----------")
by = np.sum(df_monthly['age_from_100'])
print("Total age_from_100:", f'{by:,d}')
print("Count after excl. age_from_100:", f'{TOT-by:,d}')
print(str(np.round((by/TOT) * 100,2))+'%\n')

# NANs
print("NANs\n-----------")
NANS = np.sum(df_monthly.NaNs)
print("Total NANS:", f'{NANS:,d}')
print("Count after NANS:", f'{TOT-NANS:,d}')
print(str(np.round((NANS/TOT) * 100,3))+'%\n')


# TOTAL UNIQUE REMOVED
print("TOTAL REMOVED\n-----------")
filt = np.sum(df_monthly.filtered_tripc)
print("Total removed:", f'{filt:,d}')
print("Count after removed:", f'{TOT-filt:,d}')
print(str(np.round(((TOT-filt)/TOT) * 100,3))+'%')


# ## EDA

# ### Gender - stats

# In[18]:


# Summary stats about the difference in tripcount for genders
df_monthly['difference'] = df_monthly.filtered_M.values - df_monthly.filtered_F.values
summ_gender_filt = np.round(df_monthly.describe(),2).iloc[:,-3:]
print("Summary statistics for filtered tripcount:")
summ_gender_filt


# ### Gender - pieplot

# In[80]:


# Settings
sns.set_context("paper", font_scale=1.3)
colors = sns.color_palette('pastel')

# Data
tot_male   = np.sum(df_monthly.filtered_M)
tot_female = np.sum(df_monthly.filtered_F)
tot_tripc  = np.sum(df_monthly.filtered_tripc)

# Plot
plt.pie(x=[tot_male/tot_tripc, tot_female/tot_tripc],
        labels=["Male", "Female"],
        autopct='%.1f%%', 
        colors=colors)

# Styling
#plt.legend(labels, loc='best', title="Gender")
plt.title("Percentage of Citi Bike trips in 2019 by gender");

# Save figure
plt.savefig('../../2_Thesis/images/cb-pieplot-gender-2019.png', bbox_inches='tight')


# ### Gender vs. time - lineplot

# In[81]:


# Get data
prop_male = df_monthly.filtered_M / df_monthly.filtered_tripc
prop_female = df_monthly.filtered_F / df_monthly.filtered_tripc
d = {'month': df_monthly.month, 'prop_male': prop_male, 'prop_female': prop_female}
df_temp = pd.DataFrame(data=d)

print(df_temp)


# In[88]:


# Put data in long format for easy plotting
df_temp2 = pd.melt(df_temp,id_vars=['month'],var_name='gender', value_name='bike_trips')
df_temp2['gender'].replace('prop_male', 'male', inplace=True)
df_temp2['gender'].replace('prop_female', 'female', inplace=True)

# Settings
sns.set_context("paper", font_scale=1.3)

# Plot
sns.lineplot(data=df_temp2, x="month", y="bike_trips", hue="gender", 
             style="gender", markers=True, dashes=False)

plt.title("Proportion of Citi Bike trips in 2019 per month by gender")

# x-axis
plt.xlabel("Month")
month_abs = ['','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(range(0,13), month_abs)
plt.tick_params(axis="x", rotation=50)

# y-axis
plt.ylabel("Proportion of bike trips")
plt.ylim(bottom=0, top=.9);

# Save figure
plt.savefig('../../2_Thesis/images/cb-lineplot-gender-monthly.png', bbox_inches='tight')


# ### Gender vs. location - maps

# #### Algorithm

# In[2]:


#############################
# STATION USE/MONTH BY GENDER
#############################

tot_starts = pd.DataFrame()
tot_ends = pd.DataFrame()

# Cols to extract
directory = '../Data_2019_Converted/'
cols = ['start_stat_id', 'start_stat_name', 'start_stat_lat', 'start_stat_lon', 'start_month',
        'end_stat_id', 'end_stat_name', 'end_stat_lat', 'end_stat_lon', 
        'gender']

file_list = listdir(directory)
file_list.sort()

for i in file_list:
    
    # Import file and encode gender
    df = pd.read_parquet(directory+i, columns=cols)
    df = pd.get_dummies(df, columns=['gender'])
    df.rename(columns = {'gender_0':'male', 'gender_1':'female'}, inplace = True)
    
    # Start station
    starts = df[['start_stat_id', 'male', 'female']].copy()
    
    starts_gender_count = pd.DataFrame(starts.groupby('start_stat_id').sum()[['male', 'female']])
    starts_gender_count.reset_index(inplace=True)
    
    starts = df[['start_stat_id', 'start_stat_name', 'start_stat_lat', 'start_stat_lon', 
                 'start_month']].drop_duplicates()
    starts = pd.merge(starts, starts_gender_count, on='start_stat_id')
    
    tot_starts = pd.concat([tot_starts, starts], ignore_index=True)
    del starts
    
    # End station
    ends = df[['end_stat_id', 'male', 'female']].copy() 
    
    ends_gender_count = pd.DataFrame(ends.groupby('end_stat_id').sum()[['male', 'female']])
    ends_gender_count.reset_index(inplace=True)
    
    ends = df[['end_stat_id', 'end_stat_name', 'end_stat_lat', 'end_stat_lon', 
               'start_month']].drop_duplicates()
    ends = pd.merge(ends, ends_gender_count, on='end_stat_id')
    
    tot_ends = pd.concat([tot_ends, ends], ignore_index=True)
    del ends
    del df


# In[11]:


#################################
# AVG STATION USE/MONTH BY GENDER
#################################

# Get unique start stations
cols = ['start_stat_id', 'start_stat_name', 'start_stat_lat', 'start_stat_lon']
starts_unique = tot_starts[cols].drop_duplicates()
print("Check 1: There  are", len(starts_unique), "stations.")

# Gender monthly averages
starts_monthly_avg_gender = np.round(pd.DataFrame(tot_starts.groupby('start_stat_id').sum()[['male', 'female']])/12)
starts_monthly_avg_gender.reset_index(inplace=True)
print("Check 2: There  are", len(starts_monthly_avg_gender), "stations.\n")

print(
    """
    There are duplicate stations in December 2019 due to rounding in the lat/lon. 
    Remove the two # below this piece of text in the code to get a list of the duplicates.
    Because data is already aggregated per station, the duplicates can be removed without influencing 
    the results.\n
    """)
#starts = df[['start_stat_id', 'start_stat_name', 'start_stat_lat', 'start_stat_lon', 'start_month']].drop_duplicates()
#starts[starts.duplicated(subset=['start_stat_id'])].sort_values(by=['start_stat_id'])

# Drop duplicates
starts_unique = tot_starts[cols].drop_duplicates(subset=['start_stat_id'], keep='first')
starts_unique_count = len(starts_unique)
print("Check after removal of duplicates:", starts_unique_count, 'unique stations.\n')

# Merge
starts_final = pd.merge(starts_unique, starts_monthly_avg_gender, on='start_stat_id')

# Unused start stations
stat_unused_M = np.sum(starts_final.male==0.)
stat_unused_F = np.sum(starts_final.female==0.)

print(stat_unused_M, 
      '('+str(np.round((stat_unused_M/starts_unique_count)*100,2))+'%)', 
      'stations did not see a man in 2019')

print(stat_unused_F, 
      '('+str(np.round((stat_unused_F/starts_unique_count)*100,2))+'%)', 
      'stations did not see a woman in 2019')


# In[ ]:





# In[ ]:





# #### Male cyclists map

# In[ ]:


from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import plotly.express as px
token = 'pk.eyJ1IjoidGhvcnRob3J0aG9yIiwiYSI6ImNrd2ozOWY1bTFlY24yeG5zYmgxczE0dnMifQ.aUrKUVG7NKJLC8Gp_f1ISg'


# In[301]:


##### AVG MALE CYCLISTS PER MONTH PER STATION

fig = px.scatter_mapbox(starts_final, 
                        lat="start_stat_lat", lon="start_stat_lon", hover_name="start_stat_name",
                        color_continuous_scale=px.colors.sequential.Turbo, color='male', 
                        zoom=10.8, height=660, width=600)

#fig1.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(
    mapbox = {
        'accesstoken': token,
        'style': "light"},
    showlegend = False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.write_image("../../2_Thesis/images/map-male.png")
fig.show()
del fig


# #### Female cyclists map

# In[310]:


##### AVG FEMALE CYCLISTS PER MONTH PER STATION

fig = px.scatter_mapbox(starts_final, 
                        lat="start_stat_lat", lon="start_stat_lon", hover_name="start_stat_name",
                        color_continuous_scale=px.colors.sequential.Turbo, color='female', 
                        zoom=10.8, height=660, width=600)


#fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(
    mapbox = {
        'accesstoken': token,
        'style': "light"},
    showlegend = False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.write_image("../../2_Thesis/images/map-female.png")
fig.show()
del fig


# ### Gender vs. birth year and user type - boxplot

# #### Boxplot

# In[104]:


## CREATE BOXPLOT
# This code snippet requires the sample dataset to be imported

if 'df_sample' in locals():
    
    # Subset data to increase computational speed
    cols = ['usertype', 'birthyear', 'gender']
    df_sample_box = df_sample[cols].copy()
    
    # Settings
    sns.set_style('ticks')
    sns.set_context("paper", font_scale=1.2)

    # Plot
    g = sns.catplot(y="usertype", x="birthyear",

                    hue="gender", orient='h',

                    data=df_sample_box, kind="box", facet_kws={'legend_out': False},

                    height=2.6, aspect=2., palette="pastel").set(xlabel='Birth year', ylabel='User type')

    ## Modify legend
    # Title
    new_title = 'Gender'
    g._legend.set_title(new_title)

    # Replace labels
    new_labels = ['Male', 'Female']
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)

    sns.move_legend(
        g, "upper left",
        bbox_to_anchor=(.88, .7), title=None, frameon=False,
    )

    # Styling
    plt.title("Distribution of birth year by user type and gender")
    plt.yticks([0,1], ['Subscriber', 'Customer']);
    new_y_ticks = np.linspace(1920, 2000, num=9).astype('int')
    plt.xticks(new_y_ticks)

    # Save figure
    plt.savefig('../../2_Thesis/images/boxplot-birthyear.png', bbox_inches='tight', transparent=True);

else:
    print("Error: df_sample is not imported or defined under another name")


# #### Boxplot data

# In[105]:


## GET BOXPLOT DATA
# This cell requires the subset 'df_sample_box' (see previous cell)

# Define function for retrieving boxplot data
def summ_stats(name, variable):
    
    median = np.median(variable)
    upper_quartile = np.percentile(variable, 75)
    lower_quartile = np.percentile(variable, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = variable[variable<=upper_quartile+1.5*iqr].max()
    lower_whisker = variable[variable>=lower_quartile-1.5*iqr].min()
    
    rows = ['median', 'upper_quartile', 'lower_quartile', 'iqr', 'upper_whisker', 'lower_whisker']
    values = [median, upper_quartile, lower_quartile, iqr, upper_whisker, lower_whisker]
    
    d = {'statistic':rows, name:values}
    df = pd.DataFrame(data=d)
    
    return df

if 'df_sample_box' in locals():

    # Get subsets
    male_sub = df_sample_box[(df_sample_box['gender']==0) & (df_sample_box['usertype']==0)]
    male_cus = df_sample_box[(df_sample_box['gender']==0) & (df_sample_box['usertype']==1)]
    female_sub = df_sample_box[(df_sample_box['gender']==1) & (df_sample_box['usertype']==0)]
    female_cus = df_sample_box[(df_sample_box['gender']==1) & (df_sample_box['usertype']==1)]

    # Get boxplot stats per subset
    male_sub_stats   = summ_stats('male_sub', male_sub.birthyear)
    female_sub_stats = summ_stats('female_sub', female_sub.birthyear)
    male_cus_stats   = summ_stats('male_cus', male_cus.birthyear)
    female_cus_stats = summ_stats('female_cus', female_cus.birthyear)

    # Combine in dataframe
    all_stats = male_sub_stats.merge(female_sub_stats)
    all_stats = all_stats.merge(male_cus_stats)
    all_stats = all_stats.merge(female_cus_stats)

    print(all_stats)
    
else:
    print("df_sample_box is not imported or named differently")


# #### 3-dim Frequency table

# In[99]:


if 'df_sample' in globals():

    bins = np.linspace(1920, 2010, num=10).astype('int')
    print("bins:", bins, '\n')
    df_sample['birthyear_bins'] = pd.cut(df_sample['birthyear'], bins)

    # Frequency table
    print("FREQUENCY TABLE (birth year, gender, user type)\n")
    print(np.round(
        pd.crosstab(
            df_sample.birthyear_bins,[df_sample.usertype, df_sample.gender], 
                         margins=True, normalize=True
        )*100,2
    ))
else:
    print("Error: dataframe with samples not found. Check name / if imported.")


# In[ ]:





# In[ ]:





# ## Prediction
# 
# This section requires a file with sample data from 2019 to be imported ("df_sample").

# In[3]:


def load_dataset(path):
    # load the data as a pandas dataframe
    data = pd.read_parquet(path)
    # remove irrelevant features
    cols = ['starttime', 'start_date', 'start_stat_id', 'start_stat_lat', 
            'start_stat_lon', 'stoptime',  'stop_date', 'end_stat_id', 
            'end_stat_lat', 'end_stat_lon']
    data.drop(cols, axis=1, inplace=True)
    # define target
    target_name = 'gender'
    # split data into input (X) and output (y) variables
    X = data.drop([target_name], axis=1)
    y = data[target_name]
    return X, y

def splits(X, y):
    
    # divide into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.33, 
                                                        stratify=y,
                                                        shuffle=True,
                                                        random_state=42)
    # further subdivide train into train and test
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=0.33, 
                                                      stratify=y_train,
                                                      shuffle=True,
                                                      random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# load the dataset
path = '../Data_2019_Sample/cb-2019-sample-rs-0.parquet'
X, y = load_dataset(path)
# split the dataset
X_train, X_val, X_test, y_train, y_val, y_test = splits(X, y)
# summarise
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# In[4]:


def ohe_stations(train_features, val_features, test_features):
    # define feature to ohe
    categorical_features = ["start_stat_name", "end_stat_name"]
    # initiate transformer
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=True)
    # initiate preprocessor
    preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, categorical_features)], 
    remainder='passthrough')
    # fit on train features
    preprocessor.fit(train_features)
    # transform on all feature sets
    X_train = preprocessor.transform(train_features)
    X_val = preprocessor.transform(val_features)
    X_test = preprocessor.transform(test_features)
    # get feature names
    feature_names = preprocessor.get_feature_names()
    # return the transformed data
    return X_train, X_val, X_test, feature_names

# one-hot encode the station names
X_train, X_val, X_test, feature_names = ohe_stations(X_train, X_val, X_test)
# summarise
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# ### Feature selection

# In[5]:


from sklearn.feature_selection import SelectKBest, chi2

def feat_select(no_of_features):
    # initiate selecter and fit on training data
    selector = SelectKBest(score_func=chi2, k=no_of_features).fit(X_train, y_train)
    # transform train data so that it has k features
    X_train_filt = selector.transform(X_train)
    X_val_filt = selector.transform(X_val)
    X_test_filt = selector.transform(X_test)
    # get score for each feature
    scores = selector.scores_
    # get indices of selected features
    cols = selector.get_support(indices=True) 
    # get feature names
    selected_feature_names = [feature_names[i] for i in cols] 
    
    return X_train_filt, X_val_filt, X_test_filt, scores, selected_feature_names

# select features and transform data
k=20
X_train_filt, X_val_filt, X_test_filt, scores, selected_feature_names = feat_select(k)
# summarise
print("Feature set shapes after feature selection")
print("X_train:", X_train_filt.shape)
print("X_val:", X_val_filt.shape)
print("X_test:", X_test_filt.shape)


# In[175]:


# set right input format for data
X_train_filt = X_train_filt.toarray()
X_val_filt = X_val_filt.toarray()
X_test_filt = X_test_filt.toarray()

# save data, first set directory
directory = '../Data_processed/'
# save training data
np.save(directory+'X_train.npy', X_train_filt)
np.save(directory+'y_train.npy', y_train)
# save validation data
np.save(directory+'X_val.npy', X_val_filt)
np.save(directory+'y_val.npy', y_val)
# save test data
np.save(directory+'X_test.npy', X_test_filt)
np.save(directory+'y_test.npy', y_test)


# In[202]:


# dataframe with scores per feature
d = {'feature': feature_names, 'score': scores}
df_fs = pd.DataFrame(d)
top = 20
df_fs_top = df_fs.sort_values(by='score', ascending=False)[:top]
df_fs_top = df_fs_top.sort_values(by='score',ascending=True)


# In[209]:


sns.set_context("paper", font_scale=1.6)
sns.set_style("ticks")
plt.figure(figsize=(5,7))

# plot scores
plt.barh(df_fs_top['feature'], df_fs_top['score'])
plt.title('Feature importance scores')
plt.ylabel('Feature')
plt.xlabel('Score')
# save image
plt.savefig('../../2_Thesis/images/feature-selection.png', bbox_inches='tight', transparent=True);
plt.show()


# ### Evaluation report function

# In[176]:


# function to exctract clf evaluation criteria
def classif_perf(y_true, y_pred):
    # classification report
    cr = metrics.classification_report(y_true, y_pred, target_names=['Male','Female'])
    # confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # balanced accuracy
    ba = metrics.balanced_accuracy_score(y_true, y_pred)
    return cr, cm, ba


# ### Naive Bayes
# 
# * See [mixed NB package](https://pypi.org/project/mixed-naive-bayes/)

# In[15]:


selected_feature_names


# #### Fit model

# In[177]:


# initiate classifier
nb_clf = MixedNB(categorical_features=[0,1,2,3,4,5,6,12,17])
# fit classifier
nb_clf.fit(X_train_filt, y_train)
# save model
filename = '../Models/nb_finalized_model.sav'
joblib.dump(nb_clf, filename)
# predict on val
nb_y_pred = nb_clf.predict(X_val_filt)
# extract evaluation criteria
nb_cr, nb_cm, nb_ba = classif_perf(y_val, nb_y_pred)


# In[297]:


nb_clf.get_params()


# #### Classification report

# In[179]:


# classification report
print("Classification Report")
print("----------------------")
print(nb_cr)
print()
# balanced accuracy
print("Balanced accuracy:", np.round(nb_ba, 3))


# #### Confusion matrix

# In[183]:


# Confusion matrix
tn, fp, fn, tp = nb_cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

plt.figure(figsize=(5,5))

sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for NB classifier")
# Save figure
plt.savefig('../../2_Thesis/images/cm-nb.png', bbox_inches='tight', transparent=True);


# #### ROC-AUC

# In[185]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(y_val, nb_y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='NB estimator').plot()

# title
plt.title("ROC-AUC for NB classifier")

# save figure
plt.savefig('../../2_Thesis/images/rocauc-nb.png', bbox_inches='tight', transparent=True);


# ### Decision Tree

# #### Fit model

# In[187]:


# construct search field
tree_param = {'criterion':['gini','entropy'],
              'max_depth':[5,6,7,8,9,10,15,20,30,50], 
              'class_weight': ['balanced'], 
              'random_state':[0]}
# initiate classifier
clf_dt = GridSearchCV(DecisionTreeClassifier(), tree_param)
# fit on training data
clf_dt = clf_dt.fit(X_train_filt, y_train)
# save model
filename = '../Models/dt_finalized_model.sav'
joblib.dump(clf_dt, filename)
# predict
dt_y_pred = clf_dt.predict(X_val_filt)
# extract evaluation results
dt_cr, dt_cm, dt_ba = classif_perf(y_val, dt_y_pred)

# if Python returns 'UndefinedMetricWarning' it is very likely that the either of the labels
# does not appear in the predicted labels, i.e., the classifier assigns one and only one class
# to all instances.


# In[289]:


clf_dt.best_estimator_.get_params()


# #### Classification report

# In[105]:


# Report results
print("Classification Report")
print("----------------------")
print(dt_cr)
print()
print("Balanced accuracy:", np.round(dt_ba,3))


# #### Confusion matrix

# In[125]:


# Confusion matrix
tn, fp, fn, tp = dt_cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

plt.figure(figsize=(5,5))

sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for DT classifier")
# Save figure
plt.savefig('../../2_Thesis/images/cm-dt.png', bbox_inches='tight', transparent=True);


# #### ROC-AUC

# In[126]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(y_val, dt_y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='decision tree estimator').plot()

# title
plt.title("ROC-AUC for DT classifier")

# Save figure
plt.savefig('../../2_Thesis/images/rocauc-dt.png', bbox_inches='tight', transparent=True);


# #### Save model

# In[ ]:





# ### Random Forest

# In[129]:


# construct search field
rf_param = {'n_estimators':[100, 150],
            'criterion':['gini','entropy'],
            'max_depth':[5,10,15,20,30,50,80],
            'min_samples_split':[1000],
            'class_weight': ['balanced'], 
            'random_state':[4]}
# initiate classifier
clf_rf = GridSearchCV(RandomForestClassifier(random_state=4), rf_param)
# fit on training data
clf_rf.fit(X_train_filt.toarray(), y_train.values)
# predict
y_pred = clf_rf.predict(X_val_filt.toarray())
# extract evaluation results
cr, cm, ba = classif_perf(y_val, y_pred)


# In[295]:


clf_rf.best_estimator_.get_params()


# In[132]:


# Report results
print("Classification Report")
print("----------------------")
print(cr)
print()
print("Balanced accuracy:", np.round(ba, 3))


# #### Confusion matrix

# In[165]:


# Confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for RF classifier")
# Save figure
plt.savefig('../../2_Thesis/images/cm-rf.png', bbox_inches='tight', transparent=True);


# #### ROC-AUC

# In[166]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='random forest estimator').plot();

# title
plt.title("ROC-AUC for RF classifier")

# Save figure
plt.savefig('../../2_Thesis/images/rocauc-rf.png', bbox_inches='tight', transparent=True);


# In[149]:


clf_rf.best_estimator_.get_params()


# #### Save model

# In[156]:


# save model
filename = '../Models/rf_finalized_model.sav'
joblib.dump(clf_rf, filename)


# ### MLP 2

# #### Models

# In[262]:


def mlp_model_1():

    # configure model
    model = Sequential()
    model.add(Dense(8, input_shape=(X_train.shape[1:]), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile
    opt = SGD()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


# In[263]:


def mlp_model_2():

    # configure model
    model = Sequential()
    model.add(Dense(8, input_shape=(X_train.shape[1:]), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile
    opt = Adam()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


# In[264]:


def mlp_model_3():

    # configure model
    model = Sequential()
    model.add(Dense(8, input_shape=(X_train.shape[1:]), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile
    opt = Adam()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


# #### MLP 1

# In[ ]:


# initiate model
m = 1
model = mlp_model_1()

# data
label_train = 'train'
label_test  = 'validation' 
train_x = X_train_filt
train_y = y_train
test_x  = X_val_filt
test_y  = y_val

# fit model
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), 
                    epochs=50, batch_size=100, verbose=0)

print("model finished")

# plot performance
plt.plot(history.history['accuracy'], label=label_train)
plt.plot(history.history['val_accuracy'], label=label_test)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over epochs for MLP_{}".format(m))
plt.savefig('../../2_Thesis/images/MLP-{}-progress.png'.format(m), bbox_inches='tight', transparent=True);

# predict
y_pred = np.argmax(model.predict(test_x), axis=-1)

# results
cr, cm, ba = classif_perf(test_y, y_pred)


# In[278]:


# Report results
print("Classification Report")
print("----------------------")
print(cr)
print()
print("Balanced accuracy:", np.round(ba, 3))


# In[279]:


# Confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for MLP_{} classifier".format(m))
# Save figure
plt.savefig('../../2_Thesis/images/MLP-{}-confm.png'.format(m), bbox_inches='tight', transparent=True);


# In[280]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(test_y, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='MLP_{} estimator'.format(m)).plot()

# title
plt.title("ROC curve for MLP_{}".format(m))

# save figure
plt.savefig('../../2_Thesis/images/MLP-{}-rocauc.png'.format(m), bbox_inches='tight', transparent=True);


# In[ ]:





# #### MLP 2

# In[281]:


# initiate model
m = 2
model = mlp_model_2()

# data
label_train = 'train'
label_test  = 'validation' 
train_x = X_train_filt
train_y = y_train
test_x  = X_val_filt
test_y  = y_val

# fit model
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), 
                    epochs=50, batch_size=100, verbose=0)

print("model finished")

# plot performance
plt.plot(history.history['accuracy'], label=label_train)
plt.plot(history.history['val_accuracy'], label=label_test)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over epochs for MLP_{}".format(m))
plt.savefig('../../2_Thesis/images/MLP-{}-progress.png'.format(m), bbox_inches='tight', transparent=True);

# predict
y_pred = np.argmax(model.predict(test_x), axis=-1)

# results
cr, cm, ba = classif_perf(test_y, y_pred)


# In[282]:


# Report results
print("Classification Report")
print("----------------------")
print(cr)
print()
print("Balanced accuracy:", np.round(ba, 3))


# In[283]:


# Confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for MLP_{} classifier".format(m))
# Save figure
plt.savefig('../../2_Thesis/images/MLP-{}-confm.png'.format(m), bbox_inches='tight', transparent=True);


# In[284]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(test_y, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='MLP_{} estimator'.format(m)).plot()

# title
plt.title("ROC curve for MLP_{}".format(m))

# save figure
plt.savefig('../../2_Thesis/images/MLP-{}-rocauc.png'.format(m), bbox_inches='tight', transparent=True);


# In[ ]:





# In[ ]:





# In[300]:


model.summary()


# #### MLP 3

# In[285]:


# initiate model
m = 3
model = mlp_model_3()

# data
label_train = 'train'
label_test  = 'validation' 
train_x = X_train_filt
train_y = y_train
test_x  = X_val_filt
test_y  = y_val

# fit model
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), 
                    epochs=50, batch_size=100, verbose=0)

print("model finished")

# plot performance
plt.plot(history.history['accuracy'], label=label_train)
plt.plot(history.history['val_accuracy'], label=label_test)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over epochs for MLP_{}".format(m))
plt.savefig('../../2_Thesis/images/MLP-{}-progress.png'.format(m), bbox_inches='tight', transparent=True);

# predict
y_pred = np.argmax(model.predict(test_x), axis=-1)

# results
cr, cm, ba = classif_perf(test_y, y_pred)


# In[286]:


# Report results
print("Classification Report")
print("----------------------")
print(cr)
print()
print("Balanced accuracy:", np.round(ba, 3))


# In[287]:


# Confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for MLP_{} classifier".format(m))
# Save figure
plt.savefig('../../2_Thesis/images/MLP-{}-confm.png'.format(m), bbox_inches='tight', transparent=True);


# In[288]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(test_y, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='MLP_{} estimator'.format(m)).plot()

# title
plt.title("ROC curve for MLP_{}".format(m))

# save figure
plt.savefig('../../2_Thesis/images/MLP-{}-rocauc.png'.format(m), bbox_inches='tight', transparent=True);


# In[ ]:





# In[ ]:





# In[ ]:





# ## Test data

# In[ ]:


# load models
loaded_clf_nb  = joblib.load('../Models/nb_finalized_model.sav')
loaded_clf_dt  = joblib.load('../Models/dt_finalized_model.sav')
loaded_clf_rf  = joblib.load('../Models/rf_finalized_model.sav')
loaded_clf_mlp = keras.models.load_model('../Models/mlp_finalized')


# In[315]:


# predict
y_pred_nb  = nb_clf.predict(X_test_filt)
y_pred_dt  = clf_dt.predict(X_test_filt)
y_pred_rf  = clf_rf.predict(X_test_filt)
y_pred_mlp = np.argmax(model.predict(X_test_filt), axis=-1)


# ### NB

# In[316]:


y_pred = y_pred_nb
cr, cm, ba = classif_perf(y_test, y_pred)

# Report results
print("Classification Report")
print("----------------------")
print(cr)
print()
print("Balanced accuracy:", np.round(ba, 3))
print("F1-score:", np.round(metrics.f1_score(y_test, y_pred),3))


# In[317]:


# Confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for NB classifier")
# Save figure
plt.savefig('../../2_Thesis/images/test-NB-confm.png', bbox_inches='tight', transparent=True);


# In[318]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='NB estimator').plot()

# title
plt.title("ROC curve for NB")

# save figure
plt.savefig('../../2_Thesis/images/test-NB-rocauc.png', bbox_inches='tight', transparent=True);


# ### DT

# In[319]:


y_pred = y_pred_dt
cr, cm, ba = classif_perf(y_test, y_pred)

# Report results
print("Classification Report")
print("----------------------")
print(cr)
print()
print("Balanced accuracy:", np.round(ba, 3))
print("F1-score:", np.round(metrics.f1_score(y_test, y_pred),3))


# In[320]:


# Confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for DT classifier")
# Save figure
plt.savefig('../../2_Thesis/images/test-DT-confm.png', bbox_inches='tight', transparent=True);


# In[321]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='DT estimator').plot()

# title
plt.title("ROC curve for DT")

# save figure
plt.savefig('../../2_Thesis/images/test-DT-rocauc.png', bbox_inches='tight', transparent=True);


# In[ ]:





# ### RF

# In[322]:


y_pred = y_pred_rf
cr, cm, ba = classif_perf(y_test, y_pred)

# Report results
print("Classification Report")
print("----------------------")
print(cr)
print()
print("Balanced accuracy:", np.round(ba, 3))
print("F1-score:", np.round(metrics.f1_score(y_test, y_pred),3))


# In[323]:


# Confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for RF classifier")
# Save figure
plt.savefig('../../2_Thesis/images/test-RF-confm.png', bbox_inches='tight', transparent=True);


# In[324]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='RF estimator').plot()

# title
plt.title("ROC curve for RF")

# save figure
plt.savefig('../../2_Thesis/images/test-RF-rocauc.png', bbox_inches='tight', transparent=True);


# In[ ]:





# ### MLP

# In[329]:


y_pred = y_pred_mlp
cr, cm, ba = classif_perf(y_test, y_pred)

# Report results
print("Classification Report")
print("----------------------")
print(cr)
print()
print("Balanced accuracy:", np.round(ba, 3))
print("F1-score:", np.round(metrics.f1_score(y_test, y_pred),3))


# In[330]:


# Confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')
# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male', 'female'])
plt.xticks([0.5,1.5], ['male', 'female'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for MLP_3 classifier")
# Save figure
plt.savefig('../../2_Thesis/images/test-MLP3-confm.png', bbox_inches='tight', transparent=True);


# In[331]:


# ROC
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='MLP_3 estimator').plot()

# title
plt.title("ROC curve for MLP_3")

# save figure
plt.savefig('../../2_Thesis/images/test-MLP3-rocauc.png', bbox_inches='tight', transparent=True);


# In[ ]:





# In[ ]:




