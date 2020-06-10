import pandas as pd
import numpy as np
import json
import sys
from flask_cors import CORS, cross_origin
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime


input_file = None


def transform_data(df, transform, parentId):
  tool = transform['tool']['id']
  inputs = transform['inputParameters']
  outputs = transform['outputParameters']
  params = transform['parameters']

  for col in inputs:    
    do_fill_na = True
    if col in outputs:
      col_id = outputs[col]
      if tool == 101:
        df[col_id] = normalize_dataframe(df[col], params['rolling'], params['min'], params['max'])
      elif tool == 102:
        df[col_id] = standard_dataframe(df[col], params['rolling'])
      elif tool == 103:
        df[col_id] = fisher_transform(df[col])
      elif tool == 104:
        df[col_id] = subtract_median(df[col], params['rolling'])
      elif tool == 105:
        df[col_id] = subtract_mean(df[col], params['rolling'])
      elif tool == 106:
        df[col_id] = first_diff(df[col], params['shift'])
      elif tool == 107:
        df[col_id] = percent_return(df[col], params['shift'])
      elif tool == 108:
        df[col_id] = log_return(df[col], params['shift'])
      elif tool == 109:
        df[col_id] = clip_dataframe(df[col], params['min'], params['max'])
      elif tool == 110:
        do_fill_na = False
        df[col_id] = turn_categorical(df[col])
      elif tool == 111:
        df[col_id] = turn_ranking(df[col])
      elif tool == 112:
        df[col_id] = turn_percentile(df[col], params['rolling'])
      elif tool == 113:
        df[col_id] = power_function(df[col], params['power'])
  if do_fill_na:
    return df.fillna(0)
  return df


def do_transforms(transform, df):
  if transform['target'] == True:
    return df
  Y = pd.DataFrame(index=df.index)
  if 'children' not in transform:
    return None
  for child in transform['children']:
    X = df.copy()
    X1 = transform_data(X, child, transform['id'])
    X2 =  do_transforms(child, X1)
    if X2 is not None:
      Y = pd.concat([Y, X2], axis=1)
  return Y


def train_and_test(input_file, transforms, parameters):
  algorithmType = parameters['algorithmType']
  if algorithmType == 0:
    return kmean_clustering(input_file, transforms, parameters)
  elif algorithmType == 1:
    return knn_classifier(input_file, transforms, parameters)
  return []


def knn_classifier(input_file, transforms, parameters):
  rc = input_file.shape[0]
  train_count = int(rc * 0.8)
  df_train, df_test = prepare_train(input_file, transforms, train_count)
  df_train_labels = pd.DataFrame(index=df_train.index)
  df_test_labels = pd.DataFrame(index=df_test.index)
  label = parameters['trainLabel']
  df_train_labels[label] = df_train[label]
  SHIFT = parameters['testShift']
  df_train = df_train.shift(SHIFT).fillna(0)

  df_test_labels[label] = df_test[label]
  df_test = df_test.shift(SHIFT).fillna(0)
  
  neigh = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'])
  neigh.fit(df_train, df_train_labels[label])
  df_test_result = neigh.predict(df_test[df_test.index>1])
  df_test_score = neigh.score(df_test, df_test_labels[label])
  df_train_score = neigh.score(df_train, df_train_labels[label])
  return [[df_test_labels[label].to_numpy(), df_test_result], [[df_train_score, df_test_score]]]
  

def prepare_train(input_file, transforms, train_count):
  df = input_file.copy()
  del df['Date']
  del df['Time']

  for transform in transforms:
    for tr in transforms:
      if tr['id'] != transform['id'] and 'parentId' in transform and tr['id'] == transform['parentId']:
        if 'children' not in tr:
            tr['children'] = [transform]
        else:
            tr['children'].append(transform)
  
  df = do_transforms(transforms[0], df)

  df_train = df[df.index <= train_count]
  df_test = df[df.index > train_count]
  return df_train, df_test


def kmean_clustering(input_file, transforms, parameters):
  df0 = pd.DataFrame(index=input_file.index)
  df0['Ret'] = input_file.Open.shift(-2, fill_value=0) - input_file.Open.shift(-1, fill_value=0)
  rc = df0.shape[0]
  train_count = int(rc * 0.8)
  df_train, df_test = prepare_train(input_file, transforms, train_count)
  df0_train = df0[df0.index <= train_count]
  df0_test = df0[df0.index > train_count]

  k = parameters['n_clusters']
  k_means = KMeans(n_clusters=k).fit(df_train)
  df_train['Tar'] = k_means.predict(df_train)
  df_test['Tar'] = k_means.predict(df_test)
  
  graph = [np.cumsum(df0_test['Ret'].loc[df_test['Tar'] == c].to_numpy()) for c in range(0, k)]
  metrics = [[df0_train['Ret'].loc[df_train['Tar'] == c].sum(), df0_test['Ret'].loc[df_test['Tar'] == c].sum()] for c in range(0, k) ]
  return [graph, metrics]


def upload_input_file(file):
  global input_file
  try:
    input_file = pd.read_csv (file)
    return True
  except:
    return False


def normalize_dataframe(df, length=20, min=0, max=1):
  df_cr = df.rolling(length)
  return (((df - df_cr.min()) / (df_cr.max() - df_cr.min())) * (max-min) + min).fillna(0)


def standard_dataframe(df, length=20):
  df_cr = df.rolling(length)
  return (df - df_cr.mean()) / df_cr.std().fillna(0)


def fisher_transform(df):
  # [np.log((1.0+v)/(1-v)) * .5 for v in df[col]]
  return [0 if v==1 else np.log((1.0+v)/(1-v)) * .5 for v in df]


def subtract_mean(df, length=20):
  df_cr = df.rolling(length)
  return df-df_cr.mean()


def subtract_median(df, length=20):
  df_cr = df.rolling(length)
  return df-df_cr.median()


def first_diff(df, length=1):
  return df - df.shift(length)


def percent_return(df, length=1):
  return ((df - df.shift(length))/df.shift(length)) * 100.0


def log_return(df, length=1):
  # np.log(df['Close']/df['Close'].shift(x))
  return np.log(df / (df.shift(length))).replace(np.inf, 0).replace(-np.inf, 0)


def clip_dataframe(df, min, max):
  return df.clip(min, max)


def turn_categorical(df):
  df1 = df.astype('int')
  df2 = df1 - df1.min()
  return df2.astype('category', copy=False)

def turn_ranking(df):
  return df.rank()


def turn_percentile(df, length=20):
  if length <= 0:
    return [df[x] / df[:x+1].max() * 100 for x in range(0, df.shape[0])]
  return [df[x] / df[x-length:x+1].max() * 100 for x in range(0, df.shape[0])]


def power_function(df, function):
  if function == 'square':
    return df**2
  elif function == 'cube':
    return df**3
  elif function == 'square_root':
    return df**0.5
  elif function == 'cube_root':
    return df**(1/3)
  else:
    return df**float(function)