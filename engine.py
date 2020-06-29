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
from constants import get_x_unit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from kneed import KneeLocator
from sklearn.model_selection import KFold

input_file = None


def split_param(str):
  part = str.strip().split(",")
  params = []

  for s in part:
    p = s.split("~")
    if len(p) == 1:
      params.append(int(p[0]) if p[0].isdigit() else p[0])
    elif len(p) == 2:
      params = params + list(range(int(p[0]), int(p[1])))
    elif len(p) == 3:
      params = params + list(range(int(p[0]), int(p[1]), int(p[2])))
  return params


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


def train_and_test(input_file, transforms, parameters, optimize = False):
  algorithmType = parameters['type']
  if algorithmType == 0:
    return kmean_clustering(input_file, transforms, parameters, optimize)
  elif algorithmType == 6:
    return lda_analyse(input_file, transforms, parameters)
  elif algorithmType == 5:
    return pca_analyse(input_file, transforms, parameters)
  elif algorithmType >= 1:
    if optimize:
      if algorithmType == 1:
        return knn_optimize(input_file, transforms, parameters, algorithmType)
      else:
        return grid_optimize(input_file, transforms, parameters, algorithmType)
    else:
      return knn_classifier(input_file, transforms, parameters, algorithmType)
  return []


def pca_analyse(input_file, transforms, parameters):
  df0 = pd.DataFrame(index=input_file.index)
  rc = df0.shape[0]
  train_data_set = prepare_train(input_file, transforms, rc, 0, parameters['randomSelect'], parameters['type'], parameters)
  res_data_set = []
  for [df_train, _] in train_data_set:
    k = df_train.shape[1]
    pca = PCA(n_components=k)
    pca.fit(df_train)
    
    metrics = np.array([pca.explained_variance_ratio_, pca.singular_values_])
    res_data_set.append([metrics, metrics.T])
  return res_data_set


def get_metrics(y_true, y_pred, is_classification):
  y_true = y_true[y_true.index>=1]
  return [
    r2_score(y_true, y_pred) if is_classification else accuracy_score(y_true, y_pred, normalize=True) * 100,
    mean_squared_error(y_true, y_pred) if is_classification else precision_score(y_true, y_pred, average=None, zero_division=1),
    mean_absolute_error(y_true, y_pred) if is_classification else recall_score(y_true, y_pred, average=None, zero_division=1),
    explained_variance_score(y_true, y_pred) if is_classification else f1_score(y_true, y_pred, average=None, zero_division=1)
  ]


def knn_optimize(input_file, transforms, parameters, algorithmType):
  rc = input_file.shape[0]
  train_count = int(rc * 0.8)
  if 'trainSampleCount' in parameters:
    train_count = parameters['trainSampleCount']
  
  train_shift = parameters['testShift']
  train_data_set = prepare_train(input_file, transforms, train_count, train_shift, parameters['randomSelect'], parameters['type'], parameters)
  res_data_set = []
  for df_train, df_test, df_train_labels, df_test_labels in train_data_set:
    label = parameters['trainLabel']
    df_train_target = df_train_labels[label]
    df_test_target = df_test_labels[label]

    Y = []
    X = []
    for k in parameters['n_neighbors']:
      classifier = KNeighborsClassifier(n_neighbors=k, p=parameters['P'][0], metric=parameters['metric'][0])
      classifier.fit(df_train, df_train_target)
      score = classifier.score(df_test, df_test_target)
      X.append(k)
      Y.append(score)
    kn = KneeLocator(X, Y, S=1.2, curve='convex', direction='decreasing')
    n_neighbors = round(kn.knee, 0)
    res_data_set.append([np.array([X, Y]).T, {'n_neighbors': n_neighbors, 'P': parameters['P'][0], 'metric': parameters['metric'][0]}])
  return res_data_set


def grid_optimize(input_file, transforms, parameters, algorithmType):
  rc = input_file.shape[0]
  train_count = int(rc * 0.8)
  if 'trainSampleCount' in parameters:
    train_count = parameters['trainSampleCount']
  
  train_shift = parameters['testShift']
  train_data_set = prepare_train(input_file, transforms, train_count, train_shift, parameters['randomSelect'], parameters['type'], parameters)
  res_data_set = []
  for df_train, _, df_train_labels, _ in train_data_set:
    label = parameters['trainLabel']
    df_train_target = df_train_labels[label]

    params = []
    main_param = ''
    if algorithmType == 3:
      classifier = LogisticRegression()
      main_param = 'C'
      params = ['C', 'solver', 'penalty']
    elif algorithmType == 4:
      if parameters.get('useSVR', False):
        classifier = SVR()
      else:
        classifier = SVC(random_state=parameters['random_state'][0])
      main_param = 'C'
      params = ['C', 'gamma', 'kernel', 'degree']
    else:
      continue

    param_grid = {}
    for p in params:
      param_grid[p] = parameters[p]
    gridCV = GridSearchCV(classifier, param_grid=param_grid)
    gridCV.fit(df_train, df_train_target)
    
    result = []
    k = 0
    for idx, val in enumerate(gridCV.cv_results_['mean_test_score']):
      if not np.isnan(val):
        p = gridCV.cv_results_['params'][idx]
        result.append([p[main_param], val])
        k = k + 1
    res_data_set.append([result, gridCV.best_params_])
  return res_data_set


def lda_analyse(input_file, transforms, parameters):
  df0 = pd.DataFrame(index=input_file.index)
  rc = df0.shape[0]

  train_data_set = prepare_train(input_file, transforms, rc, 0, parameters['randomSelect'], parameters['type'], parameters)
  res_data_set = []
  for [df_train, _, df_train_target, _] in train_data_set:
    k = parameters['n_components']
    lda = LinearDiscriminantAnalysis(n_components=k)
    df_new = lda.fit_transform(df_train, df_train_target[parameters['trainLabel']])
    metrics = np.array([lda.explained_variance_ratio_])
    
    res_data_set.append([df_new.T, metrics.T])
  return res_data_set


def knn_classifier(input_file, transforms, parameters, algorithmType):
  rc = input_file.shape[0]
  train_count = int(rc * 0.8)
  if 'trainSampleCount' in parameters:
    train_count = parameters['trainSampleCount']
  
  train_shift = parameters['testShift']
  train_data_set = prepare_train(input_file, transforms, train_count, train_shift, parameters['randomSelect'], parameters['type'], parameters)
  res_data_set = []
  for df_train, df_test, df_train_labels, df_test_labels in train_data_set:
    label = parameters['trainLabel']
    
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
    df_train_target = df_train_labels if multiple else df_train_labels[label]
    df_test_target = df_test_labels if multiple else df_test_labels[label]
    if algorithmType == 3:
      df_train_target = df_train_target.astype('int')
      df_test_target = df_test_target.astype('int')
    
    classifier.fit(df_train, df_train_target)

    if algorithmType == 3:
      df_train = df_train.astype('int')
      df_test = df_test.astype('int')
    df_train_result = classifier.predict(df_train)
    df_test_result = classifier.predict(df_test)

    is_regression = algorithmType == 2 or (algorithmType==4 and parameters.get('useSVR', False))
    
    df_test_score = get_metrics(df_test_target, df_test_result, is_regression)
    df_train_score = get_metrics(df_train_target, df_train_result, is_regression)
    N = get_x_unit(rc) # int(rc / 500.0)
    
    res = []
    if multiple:
      col_idx = 0
      for _ in df_test_target:
        sel = [x for x in range(df_test_result.shape[0]) if x%N == 0]
        res.append(df_test_target.to_numpy()[sel, col_idx])
        res.append(df_test_result[sel, col_idx])
        col_idx = col_idx + 1
    else:
      df_test_target = df_test_target.to_numpy()[[x for x in range(df_test_target.shape[0]) if x%N == 0]]
      df_test_result = df_test_result[[x for x in range(df_test_result.shape[0]) if x%N == 0]]
      res = [df_test_target, df_test_result]

    res_data = [np.array(res), np.array([df_train_score, df_test_score]).T]
    res_data_set.append(res_data)
  return res_data_set


def triple_barrier(df, close='Close', open='Open', low='Low', high='High', up=10, dn=10, maxhold=10):
    final_value = []
    # Enumerate over df['Close']
    for i, c in enumerate(df[close]):

        reset = 0
        # Set our vertical barrier based on trading location
        days_passed = i
        vert_barrier = min(days_passed + maxhold, i+len(df[open][i:]))
        len_before = len(final_value)
        # Enumerate up to the vertical barrier
        for j, o in enumerate(df[open][i : min(days_passed + maxhold, i+len(df[open][i:]))]):
            # If we hit bottom barrier, -1
            if o - df[low][i+j] >= dn:
                final_value.append(-1)
                break
            # Else if we hit top barrier, 1
            elif df[high][i+j] - o >= up:
                final_value.append(1)
                break
            # Else if nothing happened, keep walking forward
            elif (j+i) < vert_barrier and reset == 0:
                reset = 1
            # Else if we hit our vertical barrier, 0
            elif((j+i) >= vert_barrier-1):
                final_value.append(0)
                break
        len_after = len(final_value)
        if len_after == len_before:
            final_value.append(0)
    return final_value

def prepare_train(input_file, transforms, train_count, train_shift, random_select, algorithmType, parameters):
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

  input_filters = parameters['inputFilters']
  input_parameters = parameters['features']
  df1 = pd.DataFrame(index=df.index)
  idx = 0
  for col in input_parameters:
    if col == parameters['trainLabel']:
      df1[col] = df[col]
      continue
    if input_filters[idx]:
      df1[col] = df[col]
    idx = idx +1
  
  df_train = None
  df_test = None
  df_train_index = None
  df_test_index = None

  if 'kFold' in parameters and parameters['kFold'] != 0:
    kf = KFold(n_splits= int(parameters['kFold']))
    kf.get_n_splits(df1)
    res = []
    for train_index, test_index in kf.split(df1):
      df_train_index = train_index[train_index<df1.shape[0]-train_shift]
      df_train_index.sort()
      df_test_index = test_index[test_index<df1.shape[0]-train_shift]
      df_test_index.sort()
      df_train = df1.loc[df_train_index, :]
      df_test = df1.loc[df_test_index, :]
      
      df_train_labels = df1.loc[df_train_index+train_shift, :]
      df_test_labels = df1.loc[df_test_index+train_shift, :]

      if algorithmType == 0 or algorithmType == 5:
        return df_train, df_test

      train_label = parameters['trainLabel']
      if train_label == '':
        return [[df_train, df_test, df_train_labels, df_test_labels]]
      df2 = pd.DataFrame(index=df.index)
      if train_label == 'triple_barrier':
        df2[train_label] = triple_barrier(df)
        df2[train_label] = df2[train_label].astype('category', copy=False)
      else:
        df2[train_label] = df[train_label]
      
      df_train_labels = df2.loc[df_train_index+train_shift, :]
      df_test_labels = df2.loc[df_test_index+train_shift, :]

      res.append([df_train, df_test, df_train_labels, df_test_labels])
    return res
  elif random_select:
    df_train = df1.sample(n=train_count, random_state=1)
    df_train = df_train.sort_index(axis=0)
    df_train_index = df_train.index - train_shift
    df_train_index = df_train_index[df_train_index > 0]
    df_train = df1.loc[df_train_index, :]

    df_test_index = df1.index.difference(df_train.index) - train_shift
    df_test_index = df_test_index[df_test_index > 0]
    df_test = df1.loc[df_test_index, :]
  else:
    df_train = df1[df.index <= train_count]
    df_train_index = df_train.index - train_shift
    df_train_index = df_train_index[df_train_index > 0]
    df_train = df1.loc[df_train_index, :]

    df_test = df1[df.index > train_count]
    df_test_index = df_test.index - train_shift
    df_test_index = df_test_index[df_test_index > 0]
    df_test = df1.loc[df_test_index, :]
  
  df_train_labels = df1.loc[df_train_index+train_shift, :]
  df_test_labels = df1.loc[df_test_index+train_shift, :]

  if algorithmType == 0 or algorithmType == 5:
    return [[df_train, df_test]]

  train_label = parameters['trainLabel']
  if train_label == '':
    return [[df_train, df_test, df_train_labels, df_test_labels]]
  df2 = pd.DataFrame(index=df.index)
  if train_label == 'triple_barrier':
    df2[train_label] = triple_barrier(df)
  else:
    df2[train_label] = df[train_label]
  
  df_train_labels = df2.loc[df_train_index+train_shift, :]
  df_test_labels = df2.loc[df_test_index+train_shift, :]

  return [[df_train, df_test, df_train_labels, df_test_labels]]


def kmean_clustering(input_file, transforms, parameters, optimize):
  df0 = pd.DataFrame(index=input_file.index)
  df0['Ret'] = input_file.Open.shift(-2, fill_value=0) - input_file.Open.shift(-1, fill_value=0)
  rc = df0.shape[0]
  train_count = int(rc * 0.8)
  if 'trainSampleCount' in parameters:
    train_count = parameters['trainSampleCount']
  
  train_data_set = prepare_train(input_file, transforms, train_count, 0, parameters['randomSelect'], parameters['type'], parameters)
  res_data_set = []
  for df_train, df_test in train_data_set:
    df0_train = df0.loc[df_train.index, :]
    df0_test = df0.loc[df_test.index, :]

    n_clusters = parameters['n_clusters']
    random_state = parameters['random_state']
    init = parameters['init']

    if not optimize:
      k_means = KMeans(n_clusters=n_clusters, random_state=random_state, init=init)

    Y = []
    X = []
    if optimize:
      maxY = None
      minY = None
      for k in n_clusters:
        k_means = KMeans(n_clusters=k, random_state=random_state[0], init=init[0])
        k_means.fit(df_train)
        Y.append(float(k_means.inertia_))
        if maxY is None or k_means.inertia_ > maxY:
          maxY = k_means.inertia_
        if minY is None or k_means.inertia_ < minY:
          minY = k_means.inertia_
        X.append(k)
      Y = [(y-minY)/(maxY-minY) for y in Y]
      kn = KneeLocator(X, Y, S=1.2, curve='convex', direction='decreasing')
      n_clusters = round(kn.knee, 0)
      k_means = KMeans(n_clusters=n_clusters, random_state=random_state[0], init=init[0])

    k_means.fit(df_train)
    df_train['Tar'] = k_means.predict(df_train)
    df_test['Tar'] = k_means.predict(df_test)
    
    graph = [np.cumsum(df0_test['Ret'].loc[df_test['Tar'] == c].to_numpy()) for c in range(0, n_clusters)]
    # graph = [df_train[col].to_numpy() for col in df_test]
    # graph = [df_test.loc[df_test['Tar'] == c].to_numpy() for c in range(0, n_clusters)]
    metrics = [[df0_train['Ret'].loc[df_train['Tar'] == c].sum(), df0_test['Ret'].loc[df_test['Tar'] == c].sum()] for c in range(0, n_clusters) ]
    
    if optimize:
      res_data_set.append([np.array([X, Y]).T, {'n_clusters': n_clusters, 'init': init[0], 'random_state': random_state[0]}])
    else:
      res_data_set.append([graph, metrics])
  return res_data_set


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