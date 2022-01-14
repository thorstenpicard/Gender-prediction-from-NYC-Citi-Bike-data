#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd
import numpy as np
import joblib
import time
import os
import tempfile
from scipy.stats import skew, kurtosis
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

# plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# sklearn
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import metrics

# naive bayes
from mixed_naive_bayes import MixedNB

# distance
from geopy.distance import geodesic
from scipy.spatial import distance
from math import pi

# MLP
import tensorflow as tf
from tensorflow import keras


# ## Setup

# In[488]:


df_raw.nunique()


# In[4]:


df_raw = pd.read_csv('../samples/original/unb-sample-0.csv')
df_raw.head()


# In[6]:


df_raw['gender'] = df_raw['gender'] - 1


# In[7]:


neg, pos = np.bincount(df_raw['gender'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive (1, female): {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


# ## Cleaning

# In[9]:


# ----- DISTANCE 
# distance function
def distance(df):
    
    # start coordinates
    a = df['start station latitude'].values
    b = df['start station longitude'].values
    s = np.stack((a, b), axis = -1)

    # end coordinates
    a = df['end station latitude'].values
    b = df['end station longitude'].values
    e = np.stack((a, b), axis = -1)
    
    theta1 = np.radians(-28.904)
    theta2 = np.radians(28.904)

    ## rotation matrix
    R1 = np.array([[np.cos(theta1), np.sin(theta1)], 
                   [-np.sin(theta1), np.cos(theta1)]])
    R2 = np.array([[np.cos(theta2), np.sin(theta2)], 
                   [-np.sin(theta2), np.cos(theta2)]])

    R1_new = np.repeat([R1], df.shape[0], axis=0)
    R2_new = np.repeat([R2], df.shape[0], axis=0)

    # rotate start and end coordinates by -29 degrees
    sT = R1 @ s.T  
    eT = R1 @ e.T
    
    # coordinates of hinge point in the rotated world 
    vT = np.stack((sT[0,:], eT[1,:]))
    
    # coordinates of hinge point in the real world 
    v = R2 @ vT
    
    # collect distances (takes around 260 (4.3 min) sec per sample file)
    distances = []
    for i in range(v.shape[1]):
        d = (geodesic((s.T[0][i], s.T[1][i]), (v[0][i], v[1][i])).km + 
             geodesic((v[0][i],v[1][i]), (e.T[0][i], e.T[1][i])).km)
        distances.append(d)
        
    df['distance_unrounded'] = distances
    df['distance'] = np.round(df['distance_unrounded'], 1)
    
    # compute speed in km/h based on tripduration in seconds
    df['speed'] = np.round(df['distance'] / df['tripduration'] * 60**2, 1)
    
    # remove unrounded distances
    df.drop('distance_unrounded', axis=1, inplace=True)
    
    return df


def eT(start_time):
    
    eT_sec = time.time() - start_time
    eT_min = np.round(eT_sec / 60, 2)
    
    print("Execution time:", np.round(eT_sec, 0), "sec (" + str(eT_min) + " min).")


# In[10]:


startTime = time.time()

df_raw = distance(df_raw)

eT(startTime)


# In[11]:


def cleaning(df_original):
    
    df = df_original.copy()
    
    # change data types
    df['usertype'] = df['usertype'].astype('category')
    df['starttime'] = pd.to_datetime(df['starttime'])
    
    # subtract new time features
    df['qrtr'] = df['starttime'].dt.quarter - 1
    df['mnth'] = df['starttime'].dt.month - 1
    df['week'] = df['starttime'].dt.isocalendar().week - 1
    df['hr']   = df['starttime'].dt.hour
    df['dayofyr']   = df['starttime'].dt.dayofyear - 1
    df['dayofmnth'] = df['starttime'].dt.day - 1
    df['weekday']   = df['starttime'].dt.weekday # 0 = Monday, 6 = Sunday
    df['weekend']   = [1 if d >= 5 else 0 for d in df['weekday']]

    # compute trip duration in minutes
    df['tripduration'] = df['tripduration'] // 60

    # change data types
    df[['start station id', 'end station id', 'week']] = df[['start station id', 
                                                             'end station id', 
                                                             'week']].astype(int)

    # get geo data
    geo_data = df[['start station id', 'start station latitude', 'start station longitude', 
                   'end station id', 'end station latitude', 'end station longitude', 'gender']].copy()
    
    # drop irrelevant features
    cols = ['starttime', 'stoptime', 'bikeid',
            'start station name', 'start station latitude', 'start station longitude', 
            'end station name', 'end station latitude', 'end station longitude']
    df.drop(cols, axis = 1, inplace=True)
    
    return df, geo_data


# In[17]:


# clean data
df_cleaned, geo_data = cleaning(df_raw)
# save cleaned data
df_cleaned.to_parquet('../samples/processed/sample_0.parquet', compression=None)


# Split into folds

# In[30]:


# function for splitting the unbalanced sample
def split(df, target_name, test_size):
    
    # random state
    rs = 0
    
    # data
    X = df.copy()
    y = X.pop(target_name)

    # train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        shuffle=True,
                                                        stratify=y, 
                                                        random_state=rs)
    
    # train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=test_size, 
                                                      shuffle=True,
                                                      stratify=y_train, 
                                                      random_state=rs)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# In[31]:


X_train, X_val, X_test, y_train, y_val, y_test = split(df_cleaned, 'gender', 0.3)


# In[32]:


# function for encoding usertype
def enc_ut(X_train, X_val, X_test):

    enc = OneHotEncoder(drop='first', categories=[['Subscriber', 'Customer']])
    uts = [np.asarray(i.usertype).reshape(-1,1) for i in (X_train, X_val, X_test)]
    enc.fit(uts[0])
    
    # transform
    v = 'usertype'
    X_train[v], X_val[v], X_test[v] = [enc.transform(uts[i]).toarray().astype(int) for i in range(3)]
    
    return X_train, X_val, X_test


# In[33]:


X_train, X_val, X_test = enc_ut(X_train, X_val, X_test)


# In[34]:


def scaling(Xs, feature_names):
    
    for i in feature_names:
        sc = StandardScaler()
        data = Xs[0][i].values.reshape(-1,1)
        sc.fit(data)
        
        Xs[0][i], Xs[1][i], Xs[2][i] = [sc.transform(Xs[k][i].values.reshape(-1,1)) for k in (0,1,2)]
    
    return Xs[0], Xs[1], Xs[2]


# In[35]:


num_features = ['tripduration', 'birth year', 'distance', 'speed']
Xs = [X_train, X_val, X_test]
X_train, X_val, X_test = scaling(Xs, num_features)


# In[25]:


# function for encoding station IDs
def ohe_stations(X_train, X_val, X_test):

    # set up
    transformer = OneHotEncoder(handle_unknown="ignore", sparse=True)
    features = ["start station id", "end station id"]
    preprocessor = ColumnTransformer(transformers=[('cat', 
                                                    transformer, 
                                                    features)], 
                                     remainder='passthrough')
    # fit & transform
    preprocessor.fit(X_train)
    X_train, X_val, X_test = [preprocessor.transform(i) for i in (X_train, X_val, X_test)]
    
    return X_train, X_val, X_test, preprocessor.get_feature_names()


# In[36]:


X_train, X_val, X_test, feat_names = ohe_stations(X_train, X_val, X_test)


# In[37]:


print("train:\t", X_train.shape, y_train.shape)
print("val:\t", X_val.shape, y_val.shape)
print("test:\t", X_test.shape, y_test.shape)


# ## Feature Selection

# In[39]:


def feat_select(X, y, feat_names):
    
    # fit DT
    model = DecisionTreeClassifier(class_weight = 'balanced', random_state=0)
    model.fit(X, y)

    # feature importance based on gini
    importance = model.feature_importances_

    # match feature name with gini score
    score_list = [] # index, name, score
    for i, v in enumerate(importance):
        score_list.append([i, feat_names[i], v])

    # create df
    df_score = pd.DataFrame(score_list, columns=['index', 'name', 'gini'])
    df_score = df_score.sort_values(by=['gini'], ascending=False)
    
    return model, df_score


# In[40]:


startTime = time.time()

model_unb, df_score_unb = feat_select(X_train, y_train, feat_names)

filename = '../models/new/DT_FS_0.sav'
joblib.dump(model_unb, filename)

eT(startTime)


# In[41]:


# plot feature importance scores
sns.set_context("paper", font_scale=1.6)
sns.set_style("ticks")
plt.figure(figsize=(5,7))

k = 20
top = df_score_unb[:k].sort_values(by='gini', ascending=True)

# plot scores
plt.barh(top['name'][:20], top['gini'][:20])
plt.title('Feature importance scores based on Gini')
plt.ylabel('Feature')
plt.xlabel('Score')
# save image
plt.savefig('../images/DT_feature_selection.png', bbox_inches='tight', transparent=True);


# In[499]:


df_score_unb[:20].describe()


# ### Subsetting based on FS

# In[42]:


# select top k features
k = 10

# transform features
selector    = SelectFromModel(model_unb, prefit=True, max_features=k, threshold=-np.inf)
X_train_red = selector.transform(X_train)
X_val_red   = selector.transform(X_val)
X_test_red  = selector.transform(X_test)
feat_names_red = np.array(feat_names)[selector.get_support()]


# In[43]:


print("train red.:\t", X_train_red.shape, y_train.shape)
print("val red.:\t", X_val_red.shape, y_val.shape)
print("test red.:\t", X_test_red.shape, y_test.shape)
print("feat names:\t", feat_names_red.shape)


# ### RUS

# In[47]:


# summarize class distribution
print(Counter(y_train))

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')

# fit and apply the transform
X_over, y_over = undersample.fit_resample(X_train_red, y_train)

# summarise class distribution
print(Counter(y_over), '\n')

print("shape original train set:\t", X_train_red.toarray().shape, y_train.shape)
print("shape undersampled train set:\t", X_over.toarray().shape, y_over.shape)


# ## Evaluation function

# In[460]:


def evaluation(y_true, y_pred, y_score):
    
    # compute metrics (focus on label 1, female)
    dic = {}
    
    dic['f1'] = metrics.f1_score(y_true, y_pred)
    dic['bal_acc'] = metrics.balanced_accuracy_score(y_true, y_pred)
    dic['precision'] = metrics.precision_score(y_true, y_pred)
    dic['recall'] = metrics.recall_score(y_true, y_pred)
                            
    dic['roc_auc'] = metrics.roc_auc_score(y_true, y_score)
    dic['pr_auc']  = metrics.average_precision_score(y_true, y_score)
    
    dic['cm'] = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = dic['cm'].ravel()
    
    dic
    dic['sensitivity (tpr, rec)'] = tp / (tp + fn) # how well the positive class is predicted
    dic['specificity (tnr)'] = tn / (fp + tn) # how well the negative class is predicted
    
    return dic


# ## Naive Bayes

# In[44]:


# get indices of categorical columns
spm = X_train_red
fn = feat_names_red

df_temp = pd.DataFrame.sparse.from_spmatrix(spm)
df_temp = pd.DataFrame({'index': df_temp.columns.values, 'name': fn})
num_features = ['tripduration', 'birth year', 'distance', 'speed']
cat_indices = df_temp[~df_temp['name'].isin(num_features)].index.values


# ### NB original

# **Validation**

# In[462]:


model_NB = MixedNB(categorical_features=cat_indices)
model_NB.fit(X_train_red, y_train)


# In[463]:


y_pred = model_NB.predict(X_val_red)
NB_scores = evaluation(y_val, y_pred, y_pred)
NB_scores


# In[ ]:


# save model
filename = '../models/final/NB_model.sav'
joblib.dump(model_NB, filename)


# **Test**

# In[464]:


y_pred = model_NB.predict(X_test_red)
NB_test_scores = evaluation(y_test, y_pred, y_pred)
NB_test_scores


# ### NB RUS

# **Validation**

# In[465]:


model_NB_RUS = MixedNB(categorical_features=cat_indices)
model_NB_RUS.fit(X_over, y_over)


# In[466]:


y_pred = model_NB_RUS.predict(X_val_red)
NB_RUS_val_scores = evaluation(y_val, y_pred, y_pred)
NB_RUS_val_scores


# In[ ]:


# save model
filename = '../models/final/NB_RUS_model.sav'
joblib.dump(model_NB_RUS, filename)


# **Test**

# In[467]:


y_pred = model_NB_RUS.predict(X_test_red)
NB_RUS_test_scores = evaluation(y_test, y_pred, y_pred)
NB_RUS_test_scores


# In[ ]:





# ## Decision Tree

# ### Set up

# In[107]:


def DT(X, y, cw):
    grid = {'criterion':['gini', 'entropy'],
            'max_depth':[10, 15, 30, 50], 
            'class_weight': [cw],
            'min_samples_leaf': [10, 100],
            'min_samples_split': [100],
            'random_state':[0]}
    
    # initiate and fit
    model = GridSearchCV(DecisionTreeClassifier(), grid, scoring='f1', cv=2, verbose=1)
    model = model.fit(X, y)

    return model

def pred(model, X):
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, y_score


# ### DT original

# In[108]:


startTime = time.time()
model_DT1 = DT(X_train_red, y_train, None)
model_DT2 = DT(X_train_red, y_train, 'balanced')
eT(startTime)


# #### DT1

# In[109]:


# obtain scores DT1
y_pred, y_score = pred(model_DT1, X_val_red)
DT1_val_scores = evaluation(y_val, y_pred, y_score)
DT1_val_scores


# In[110]:


model_DT1.best_estimator_.get_params()


# In[119]:


# save model
filename = '../models/final/DT1_model.sav'
joblib.dump(model_DT1, filename)


# In[469]:


y_pred, y_score = pred(model_DT1, X_test_red)
DT1_test_scores = evaluation(y_test, y_pred, y_score)
DT1_test_scores


# #### DT2

# In[111]:


# obtain scores DT2
y_pred, y_score = pred(model_DT2, X_val_red)
DT2_val_scores = evaluation(y_val, y_pred, y_score)
DT2_val_scores


# In[112]:


model_DT2.best_estimator_.get_params()


# In[120]:


# save model
filename = '../models/final/DT2_model.sav'
joblib.dump(model_DT2, filename)


# #### Test

# In[470]:


y_pred, y_score = pred(model_DT2, X_test_red)
DT2_test_scores = evaluation(y_test, y_pred, y_score)
DT2_test_scores


# In[505]:


# Confusion matrix
cm = DT2_test_scores['cm']
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix")
print("----------------------")
print("tn:", f'{tn:,d}')
print("fp:", f'{fp:,d}')
print("fn:", f'{fn:,d}')
print("tp:", f'{tp:,d}')

# Plot confusion matrix
sns.set_context("paper", font_scale=1.6)
plt.figure(figsize=(6,6))
sns.heatmap(data=cm,linewidths=.5, annot=True, fmt='d',square=True, cmap='Blues')

# set labels and title
plt.ylabel('Actual label')
plt.yticks([0.5,1.5], ['male (0)', 'female (1)'])
plt.xticks([0.5,1.5], ['male (0)', 'female (1)'])
plt.xlabel('Predicted label')
plt.title("Confusion matrix for the DT \nclassifier in the weighted strategy")
# Save figure
plt.savefig('../images/CM_DT_Weighted.png', bbox_inches='tight', transparent=True);


# In[ ]:





# In[ ]:





# ### DT RUS

# #### DT1

# In[114]:


startTime = time.time()
model_DT1_RUS = DT(X_over, y_over, None)
model_DT2_RUS = DT(X_over, y_over, 'balanced')
eT(startTime)


# In[115]:


# obtain scores DT1
y_pred, y_score = pred(model_DT1_RUS, X_val_red)
DT1_RUS_val_scores = evaluation(y_val, y_pred, y_score)
DT1_RUS_val_scores


# In[471]:


# obtain scores DT1
y_pred, y_score = pred(model_DT1_RUS, X_test_red)
DT1_RUS_test_scores = evaluation(y_test, y_pred, y_score)
DT1_RUS_test_scores


# In[116]:


model_DT1_RUS.best_estimator_.get_params()


# In[121]:


# save model
filename = '../models/final/DT1_RUS_model.sav'
joblib.dump(model_DT1_RUS, filename)


# #### DT2

# In[122]:


# save model
filename = '../models/final/DT2_RUS_model.sav'
joblib.dump(model_DT2_RUS, filename)


# #### Test

# In[118]:


y_pred, y_score = pred(model_DT1_RUS, X_test_red)
DT1_RUS_test_scores = evaluation(y_test, y_pred, y_score)
DT1_RUS_test_scores


# ## MLP

# In[241]:


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[242]:


train_features = X_train_red.toarray()
val_features = X_val_red.toarray()
test_features = X_test_red.toarray()
train_features_rus = X_over.toarray()

train_labels = y_train.copy()
val_labels = y_val.copy()
test_labels = y_test.copy()
train_labels_over = y_over.copy()


# In[257]:


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(
            16, activation='relu',
            input_shape=(X_train_red.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model


# In[258]:


EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


# In[259]:


model = make_model()
model.summary()


# In[260]:


model.predict(train_features[:10])


# In[261]:


results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))


# #### Set bias

# In[262]:


initial_bias = np.log([pos/neg])
initial_bias


# In[263]:


model = make_model(output_bias=initial_bias)
model.predict(train_features[:10])


# In[266]:


results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))


# In[267]:


initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)


# #### Check if bias works

# In[268]:


model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])

# zero bias 
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)


# In[269]:


model = make_model()
model.load_weights(initial_weights)

# adjusted bias
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)


# In[270]:


def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss');


# In[271]:


plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)


# From the plot above it can be derived that the adjusted bias helps the network right from the beginning.

# ### Train Model

# In[272]:


model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=20,
    callbacks=[early_stopping],
    validation_data=(val_features, y_val))


# ### Visualise training history

# In[273]:


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

    plt.legend()


# In[274]:


plot_metrics(baseline_history)


# In[275]:


train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)


# In[276]:


def plot_cm(labels, predictions, p=0.5):
    cm = metrics.confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Male cyclists Detected (True Negatives): ', cm[0][0])
    print('Male cyclists Incorrectly Detected (False Positives): ', cm[0][1])
    print('Female cyclists Missed (False Negatives): ', cm[1][0])
    print('Female cyclists Detected (True Positives): ', cm[1][1])
    print('Total female cyclists: ', np.sum(cm[1]))


# In[277]:


baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline)


# In[278]:


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal');


# In[279]:


plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')


# In[282]:


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


# In[283]:


plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')


# ## Class weights

# In[284]:


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


# In[285]:


weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels),
    class_weight=class_weight) # class weights


# In[286]:


plot_metrics(weighted_history)


# In[287]:


train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)


# In[288]:


weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)


# More false positives, lower accuracy, but higher recall and AUC compared to first model. This means that the weighted model is better at classifying the positive instances.

# In[289]:


plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right')


# In[293]:




name = 'weighted'

fp, tp, _ = metrics.roc_curve(test_labels, test_predictions_weighted)

plt.plot(100*fp, 100*tp, label=name, linewidth=2)
plt.xlabel('False positives [%]')
plt.ylabel('True positives [%]')
plt.xlim([-0.5,20])
plt.ylim([80,100.5])
plt.grid(True)
ax = plt.gca()
ax.set_aspect('equal');


# In[297]:


fp * 100


# In[298]:


plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_prc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_prc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right')


# ## MLP Baseline

# In[361]:


# MLP
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[387]:


X_train_red = X_train_red.toarray()
X_val_red = X_val_red.toarray()
X_test_red = X_test_red.toarray()
X_over = X_over.toarray()


# In[386]:


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def make_model(lr, metrics=METRICS):

    model = keras.Sequential([keras.layers.Dense(16, 
                                                 activation='relu', 
                                                 input_shape=(X_train_red.shape[-1],)),
                              keras.layers.Dense(16, 
                                                 activation='relu'),
                              keras.layers.Dense(1, 
                                                 activation='sigmoid',)
                             ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=metrics
                 )

    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=7,
    mode='max',
    restore_best_weights=True)


# In[426]:


baseline_model = make_model(1e-3)

baseline_history = baseline_model.fit(
    X_train_red,
    y_train,
    batch_size=2048,
    epochs=20,
    callbacks=[early_stopping],
    validation_data=(X_val_red, y_val), 
    verbose=0)


# In[446]:


# evaluate
loss,tp,fp,tn,fn,precision,recall,auc,prc = baseline_model.evaluate(X_val_red, y_val, verbose=0)


# In[447]:


# compute scores
tpr = tp / (tp + fn) # sensitivity
tnr = tn / (tn + fp) # specificity
f1 = 2 * (precision * recall) / (precision + recall)
ba = 1/2 * (tpr + tnr)

print("Baseline MLP val scores")
print("------------")
print("loss:", loss)
print("f1:", f1)
print("ba:", ba)
print("precision:", precision)
print("recall:", recall)
print("tpr:", tpr)
print("tnr:", tnr)
print("prc:", prc)

y_score = baseline_model.predict(X_val_red)
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

print("roc_auc:", roc_auc)


# In[442]:


plt.figure()
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=1,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic Baseline MLP model on val set")
plt.legend(loc="lower right")
plt.show()


# #### Test

# In[473]:


BATCH_SIZE


# In[480]:


baseline_results = baseline_model.evaluate(
    test_features, 
    test_labels, 
    batch_size=BATCH_SIZE, 
    verbose=0)

for name, value in zip(baseline_model.metrics_names, baseline_results):
    print(name, ': ', value)


# In[481]:


tp, fp, tn, fn = 32, 51, 134283, 45634

precision = tp / (tp + fp)
recall = tp / (tp + fn)
tpr = tp / (tp + fn) # sensitivity
tnr = tn / (tn + fp) # specificity
f1 = 2 * (precision * recall) / (precision + recall)
ba = 1/2 * (tpr + tnr)

print("f1:", f1)
print("ba:", ba)
print("sensitivity (tpr):", tpr)
print("specificity (tnr):", tnr)


# In[ ]:


plt.figure()
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=1,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic Baseline MLP model on val set")
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# ## MLP Weighted

# In[443]:


counts = np.bincount(y_train)
weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]
class_weight = {0: weight_for_0, 1: weight_for_1}

weighted_model = make_model(1e-3)

weighted_history = weighted_model.fit(
    X_train_red,
    y_train,
    batch_size=2048,
    epochs=20,
    callbacks=[early_stopping],
    validation_data=(X_val_red, y_val), 
    verbose=0, 
    class_weight=class_weight)


# In[448]:


# evaluate
loss,tp,fp,tn,fn,precision,recall,auc,prc = weighted_model.evaluate(X_val_red, y_val, verbose=0)


# In[449]:


# compute scores
tpr = tp / (tp + fn) # sensitivity
tnr = tn / (tn + fp) # specificity
f1 = 2 * (precision * recall) / (precision + recall)
ba = 1/2 * (tpr + tnr)

print("Weighted MLP val scores")
print("------------")
print("loss:", loss)
print("f1:", f1)
print("ba:", ba)
print("precision:", precision)
print("recall:", recall)
print("tpr:", tpr)
print("tnr:", tnr)
print("prc:", prc)

y_score = weighted_model.predict(X_val_red)
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

print("roc_auc:", roc_auc)


# In[450]:


plt.figure()
plt.plot(
    fpr,
    tpr,
    lw=1,
    color="darkorange",
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic weighted MLP")
plt.legend(loc="lower right")
plt.show()


# In[482]:


weighted_results = weighted_model.evaluate(
    test_features, 
    test_labels, 
    batch_size=BATCH_SIZE, 
    verbose=0)

for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)


# In[485]:


tp, fp, tn, fn = 24204, 51483, 82851, 21462

precision = tp / (tp + fp)
recall = tp / (tp + fn)
tpr = tp / (tp + fn) # sensitivity
tnr = tn / (tn + fp) # specificity
f1 = 2 * (precision * recall) / (precision + recall)
ba = 1/2 * (tpr + tnr)

print("f1:", f1)
print("ba:", ba)
print("sensitivity (tpr):", tpr)
print("specificity (tnr):", tnr)


# ## MLP RUS

# In[456]:


RUS_model = make_model(1e-3)

RUS_history = RUS_model.fit(
    X_over,
    y_over,
    batch_size=2048,
    epochs=20,
    callbacks=[early_stopping],
    validation_data=(X_val_red, y_val), 
    verbose=0)


# In[457]:


# evaluate
loss,tp,fp,tn,fn,precision,recall,auc,prc = RUS_model.evaluate(X_val_red, y_val, verbose=0)


# In[458]:


# compute scores
tpr = tp / (tp + fn) # sensitivity
tnr = tn / (tn + fp) # specificity
f1 = 2 * (precision * recall) / (precision + recall)
ba = 1/2 * (tpr + tnr)

print("RUS MLP val scores")
print("------------")
print("loss:", loss)
print("f1:", f1)
print("ba:", ba)
print("precision:", precision)
print("recall:", recall)
print("tpr:", tpr)
print("tnr:", tnr)
print("prc:", prc)

y_score = RUS_model.predict(X_val_red)
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_score, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

print("roc_auc:", roc_auc)


# In[459]:


plt.figure()
plt.plot(
    fpr,
    tpr,
    lw=1,
    color="darkorange",
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic RUS MLP")
plt.legend(loc="lower right")
plt.show()


# In[486]:


RUS_results = RUS_model.evaluate(
    test_features, 
    test_labels, 
    batch_size=BATCH_SIZE, 
    verbose=0)

for name, value in zip(RUS_model.metrics_names, RUS_results):
    print(name, ': ', value)


# In[487]:


tp, fp, tn, fn = 25740, 55685, 78649, 19926

precision = tp / (tp + fp)
recall = tp / (tp + fn)
tpr = tp / (tp + fn) # sensitivity
tnr = tn / (tn + fp) # specificity
f1 = 2 * (precision * recall) / (precision + recall)
ba = 1/2 * (tpr + tnr)

print("f1:", f1)
print("ba:", ba)
print("sensitivity (tpr):", tpr)
print("specificity (tnr):", tnr)


# In[ ]:




