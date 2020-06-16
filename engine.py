import pandas as pd
import numpy as np
import json
import sys
from flask_cors import CORS, cross_origin
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error


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
        df[col_id] = clip_dataframe(df[col], params['min'], params['max'], params['scale'])
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
  Y = pd.DataFrame(index=df.index)
  if 'children' not in transform:
    return df
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
  elif algorithmType == 5:
    return pca_analyse(input_file, transforms, parameters)
  elif algorithmType >= 1:
    return knn_classifier(input_file, transforms, parameters, algorithmType)
  return []


def pca_analyse(input_file, transforms, parameters):
  df0 = pd.DataFrame(index=input_file.index)
  rc = df0.shape[0]
  [df_train, _] = prepare_train(input_file, transforms, rc, parameters['algorithmType'])

  k = df_train.shape[1]
  pca = PCA(n_components=k)
  pca.fit(df_train)
  
  metrics = np.array([pca.explained_variance_ratio_, pca.singular_values_])
  return [metrics, metrics.T]


def get_metrics(y_true, y_pred, is_classification):
  y_true = y_true[y_true.index>1]
  return [
    r2_score(y_true, y_pred) if is_classification else accuracy_score(y_true, y_pred, normalize=True) * 100,
    mean_squared_error(y_true, y_pred) if is_classification else precision_score(y_true, y_pred, average=None, zero_division=1),
    mean_absolute_error(y_true, y_pred) if is_classification else recall_score(y_true, y_pred, average=None, zero_division=1),
    explained_variance_score(y_true, y_pred) if is_classification else f1_score(y_true, y_pred, average=None, zero_division=1)
  ]

def knn_classifier(input_file, transforms, parameters, algorithmType):
  rc = input_file.shape[0]
  train_count = int(rc * 0.8)
  df_train, df_test, df_train_labels, df_test_labels = prepare_train(input_file, transforms, train_count, parameters['algorithmType'])
  label = parameters['trainLabel']
  SHIFT = parameters['testShift']
  df_train_org = df_train.copy()
  df_test_org = df_test.copy()
  df_train = df_train.shift(SHIFT).fillna(0)
  df_test = df_test.shift(SHIFT).fillna(0)
  
  if algorithmType == 1:
    classifier = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'])
  elif algorithmType == 2:
    classifier = LinearRegression()
  elif algorithmType == 3:
    classifier = LogisticRegression(random_state=0, solver=parameters.get('solver', 'lbfgs'), penalty=parameters.get('penalty', 'l2'))
  elif algorithmType == 4:
    if parameters.get('useSVR', False):
      model = SVR(gamma=parameters.get('gamma', 'auto'), kernel=parameters.get('kernel', 'rbf'), degree=parameters.get('degree', 3))
    else:
      model = SVC(gamma=parameters.get('gamma', 'auto'), kernel=parameters.get('kernel', 'rbf'), degree=parameters.get('degree', 3))
    classifier = make_pipeline(StandardScaler(), model)
  elif algorithmType == 6:
    classifier = LinearDiscriminantAnalysis()
  multiple = parameters.get('multiple', False) and algorithmType == 2
  df_train_target = df_train_org if multiple else df_train_labels[label]
  df_test_target = df_test_org if multiple else df_test_labels[label]
  if algorithmType == 3:
    df_train_target = df_train_target.astype('int')
    df_test_target = df_test_target.astype('int')
  classifier.fit(df_train, df_train_target)

  if algorithmType == 3:
    df_train = df_train.astype('int')
    df_test = df_test.astype('int')
  df_train_result = classifier.predict(df_train[df_train.index>1])
  df_test_result = classifier.predict(df_test[df_test.index>1])

  is_regression = algorithmType == 2 or algorithmType == 3 or (algorithmType==4 and parameters.get('useSVR', False))
  
  df_test_score = get_metrics(df_test_target, df_test_result, is_regression)
  df_train_score = get_metrics(df_train_target, df_train_result, is_regression)
  N = int(rc / 100.0)
  # df_test_target = df_test_target[df_test_target.index%N == 0]
  # df_test_result = df_test_result[df_test_result.index%N == 0]
  res = []
  if multiple:
    col_idx = 0
    for _ in df_test_target:
      sel = [x for x in range(df_test_result.shape[0]) if x%N == 0]
      res.append(df_test_target.to_numpy()[sel, col_idx])
      res.append(df_test_result[sel, col_idx])
      col_idx = col_idx + 1
    # df_test_result = df_test_result[[x for x in range(df_test_result.shape[0]) if x%N == 0], 0]
    # df_test_target = df_test_target.to_numpy().T
    # df_test_result = df_test_result[[x for x in range(df_test_result.shape[0]) if x%N == 0], :].T
  else:
    # df_test_target = df_test_target[df_test_target.index%N == 0].to_numpy()
    df_test_target = df_test_target.to_numpy()[[x for x in range(df_test_target.shape[0]) if x%N == 0]]
    df_test_result = df_test_result[[x for x in range(df_test_result.shape[0]) if x%N == 0]]
    res = [df_test_target, df_test_result]

  return [np.array(res), np.array([df_train_score, df_test_score]).T]
  

def prepare_train(input_file, transforms, train_count, algorithmType):
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

  input_filters = transforms[1]['inputFilters']
  input_parameters = transforms[1]['inputParameters']
  df1 = pd.DataFrame(index=df.index)
  idx = 0
  for col in input_parameters:
    if input_filters[idx]:
      df1[col] = df[col]
    idx = idx +1
  
  df_train = df1[df.index <= train_count]
  df_test = df1[df.index > train_count]

  if algorithmType == 0 or algorithmType == 5:
    return df_train, df_test

  train_label = transforms[1]['parameters']['trainLabel']
  if train_label == '':
    return df_train, df_test, None, None
  df2 = pd.DataFrame(index=df.index)
  df2[train_label] = df[train_label]
  

  df_train_labels = df2[df.index <= train_count]
  df_test_labels = df2[df.index > train_count]

  return df_train, df_test, df_train_labels, df_test_labels


def kmean_clustering(input_file, transforms, parameters):
  df0 = pd.DataFrame(index=input_file.index)
  df0['Ret'] = input_file.Open.shift(-2, fill_value=0) - input_file.Open.shift(-1, fill_value=0)
  rc = df0.shape[0]
  train_count = int(rc * 0.8)
  df_train, df_test = prepare_train(input_file, transforms, train_count, parameters['algorithmType'])
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


def clip_dataframe(df, min, max, scale=1):
  return df.clip(min, max) * scale


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