import pandas as pd
import numpy as np
import json
import sys
from flask_cors import CORS, cross_origin
from sklearn import preprocessing
from sklearn.cluster import KMeans
from datetime import datetime


input_file = None
input_data = None


def transform_data(X0, transform):
  tool = transform['tool']['id']
  if tool == 101:
    params = transform['parameters']
    return normalize_dataframe(X0, transform['outputParameters'], params['rolling'], params['min'], params['max'])
  elif tool == 102:
    params = transform['parameters']
    return standard_dataframe(X0, transform['outputParameters'], params['rolling'])
  return X0


def do_transforms(transform, df):
  if transform['target'] == True:
    return df
  Y = pd.DataFrame(index=df.index)
  for child in transform['children']:
    X = df.copy()
    X1 = transform_data(X, child)
    X2 =  do_transforms(child, X1)
    Y = pd.concat([Y, X2], axis=1)
  return Y


def train(transforms, parameters):
  df = copy_dataframe()
  df0 = pd.DataFrame(index=df.index)
  df0['Ret'] = df.Open.shift(-2, fill_value=0) - df.Open.shift(-1, fill_value=0)
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
  df = pd.concat([df, df0], axis=1)

  rc = df.shape[0]
  rc_train = int(rc * 0.8)
  df_train = df[df.index <= rc_train]
  df_test = df[df.index > rc_train]

  k = parameters['n_clusters']
  k_means = KMeans(n_clusters=k).fit(df_train)
  df_train['Tar'] = k_means.predict(df_train)
  df_test['Tar'] = k_means.predict(df_test)
  
  graph = [np.cumsum(df_test['Ret'].loc[df_test['Tar'] == c].to_numpy()) for c in range(0, k)]
  metrics = [[df_train['Ret'].loc[df_train['Tar'] == c].sum(), df_test['Ret'].loc[df_test['Tar'] == c].sum()] for c in range(0, k) ]
  return [graph, metrics]


def upload_input_file(file):
  global input_data
  global input_file
  try:
    input_file = pd.read_csv (file)
    input_data = input_file.to_numpy()
    return True
  except:
    return False


def copy_dataframe():
  global input_file
  return input_file.copy()


def normalize_dataframe(df, filters, rolling, min, max):
  for col in df.columns:
    if col == 'Date' or col == 'Time':
      continue
    if col in filters:
      df_cr = df[col].rolling(rolling, min_periods=0, center=True)
      df[col] = ((df[col] - df_cr.min()) / (df_cr.max() - df_cr.min())) * (max-min) + min
    elif col == 'Ret':
      continue
    else:
      del df[col]
  return df.fillna(0)


def standard_dataframe(df, filters, rolling):
  for col in df.columns:
    if col == 'Date' or col == 'Time':
      continue
    if col in filters:
      df_cr = df[col].rolling(rolling, min_periods=0, center=True)
      df[col] = (df[col] - df_cr.mean()) / df_cr.std()
    elif col == 'Ret':
      continue
    else:
      del df[col]
  return df.fillna(0)