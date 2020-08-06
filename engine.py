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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble  import RandomForestClassifier
from sklearn.ensemble  import RandomForestRegressor
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
from sklearn.metrics import confusion_matrix
from constants import get_x_unit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from kneed import KneeLocator
from sklearn.model_selection import KFold
from skimage import measure
from sklearn.utils import resample
from transformations import do_transforms
from preprocessor import prepare_train_test

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
  train_data_set = prepare_train_test(input_file, transforms, parameters)
  res_data_set = []
  for [df_train, _] in train_data_set:
    k = df_train.shape[1]
    pca = PCA(n_components=k)
    pca.fit(df_train)
    
    metrics = np.array([pca.explained_variance_ratio_*100, pca.singular_values_])
    res_data_set.append([metrics, metrics.T])
  return res_data_set


def get_metrics(y_true, y_pred, is_regression, train_shift, algorithmType):
  # y_true = y_true[y_true.index>=train_shift]
  cm = []
  if algorithmType != 2 and not is_regression:
    cm = confusion_matrix(y_true, y_pred)
  return [
    r2_score(y_true, y_pred) if is_regression else accuracy_score(y_true, y_pred, normalize=True) * 100,
    mean_squared_error(y_true, y_pred) if is_regression else precision_score(y_true, y_pred, average=None, zero_division=1) * 100,
    mean_absolute_error(y_true, y_pred) if is_regression else recall_score(y_true, y_pred, average=None, zero_division=1) * 100,
    explained_variance_score(y_true, y_pred) if is_regression else f1_score(y_true, y_pred, average=None, zero_division=1) * 100
  ], cm


def knn_optimize(input_file, transforms, parameters, algorithmType):
  train_data_set = prepare_train_test(input_file, transforms, parameters)
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
  train_data_set = prepare_train_test(input_file, transforms, parameters)
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
    elif algorithmType == 7:
      main_param = 'max_depth'
      params = ['max_depth', 'random_state']
      classifier = DecisionTreeClassifier()
    elif algorithmType == 8:
      main_param = 'max_depth'
      params = ['max_depth', 'random_state']
      classifier = RandomForestClassifier()
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
  train_data_set = prepare_train_test(input_file, transforms, parameters)
  res_data_set = []
  for [df_train, _, df_train_target, _] in train_data_set:
    k = parameters['n_components']
    lda = LinearDiscriminantAnalysis(n_components=k)
    df_new = lda.fit_transform(df_train, df_train_target[parameters['trainLabel']])
    metrics = np.array([lda.explained_variance_ratio_*100])

    date_index = input_file.loc[df_train.index, 'Date']
    res_data_set.append([np.concatenate(([date_index], df_new.T), axis=0), metrics.T])
  return res_data_set


def knn_classifier(input_file, transforms, parameters, algorithmType):
  train_data_set = prepare_train_test(input_file, transforms, parameters)
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
      classifier = model #make_pipeline(StandardScaler(), model)
    elif algorithmType == 6:
      classifier = LinearDiscriminantAnalysis()
    elif algorithmType == 7:
      if not parameters.get('regression', False):
        classifier = DecisionTreeClassifier(max_depth=parameters.get('max_depth', 2), random_state=parameters.get('random_state', 0))
      else:
        classifier = DecisionTreeRegressor(max_depth=parameters.get('max_depth', 2), random_state=parameters.get('random_state', 0))
    elif algorithmType == 8:
      if not parameters.get('regression', False):
        classifier = RandomForestClassifier(max_depth=parameters.get('max_depth', 2), random_state=parameters.get('random_state', 0))
      else:
        classifier = RandomForestRegressor(max_depth=parameters.get('max_depth', 2), random_state=parameters.get('random_state', 0))
      
    df_train_target = df_train_labels[label]
    df_test_target = df_test_labels[label]
    is_regression = algorithmType == 2 or (algorithmType==4 and parameters.get('useSVR', False)) or (parameters.get('regression', False))

    if label != 'triple_barrier' and not is_regression:
      df_train_target = df_train_target.astype('int').astype('category')
      df_test_target = df_test_target.astype('int').astype('category')
    
    for idx, col in enumerate(parameters['features']):
      if col == label and parameters['inputFilters'][idx] == False:
        del df_train[label]
        del df_test[label]
        break
    
    classifier.fit(df_train, df_train_target)

    # if algorithmType == 3 or algorithmType == 4:
    #   df_train = df_train.astype('int').astype('category')
    #   df_test = df_test.astype('int').astype('category')
    df_train_result = classifier.predict(df_train)
    df_test_result = classifier.predict(df_test)
    
    test_shift = parameters['testShift'] if parameters['trainLabel'] != 'triple_barrier' else 0
    df_test_score, df_test_cm = get_metrics(df_test_target, df_test_result, is_regression, test_shift, algorithmType)
    df_train_score, _ = get_metrics(df_train_target, df_train_result, is_regression, test_shift, algorithmType)
    N = 1 # int(rc / 500.0)
    
    date_index = input_file.loc[df_test.index+test_shift, input_file.columns[0]]
    date_index = date_index.fillna(date_index.max())
    res = [np.array(date_index) ]
    
    df_test_target = df_test_target.to_numpy()[[x for x in range(df_test_target.shape[0]) if x%N == 0]]
    df_test_result = df_test_result[[x for x in range(df_test_result.shape[0]) if x%N == 0]]
    res.append(df_test_target)
    res.append(df_test_result)
    
    contours, features = [[], []]
    if not is_regression and df_test.shape[1] == 2: #  or (label != 'triple_barrier' and df_test.shape[1] == 3)
      contours, features = get_decision_boundaries(classifier, df_test.values, df_test_target, 100)
    if is_regression and df_test.shape[1] == 1:
      features = np.array([[df_train[df_train.columns[0]].values, df_train_target.values], [df_test[df_test.columns[0]].values, df_test_target]])
      contours = np.array([[df_train[df_train.columns[0]].values, df_train_result], [df_test[df_test.columns[0]].values, df_test_result]])
      # contours, features = get_decision_boundaries(classifier, df_train.values, df_train_target, 100)

    res = np.array(res)
    res_data = [res, np.array([df_train_score, df_test_score]).T, df_test_cm, contours, features]
    res_data_set.append(res_data)
  return res_data_set


def triple_barrier(df, up=10, dn=10, maxhold=10, close='Close', high='High', open='Open', low='Low'):
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

# def prepare_train_test(input_file, transforms, train_count, train_shift, random_select, algorithmType, parameters):
#   df = input_file.copy()
#   if 'Date' in input_file:
#     del df['Date']
#   if 'Time' in input_file:
#     del df['Time']

#   for transform in transforms:
#     for tr in transforms:
#       if tr['id'] != transform['id'] and 'parentId' in transform and tr['id'] == transform['parentId']:
#         if 'children' not in tr:
#             tr['children'] = [transform]
#         else:
#             tr['children'].append(transform)
  
#   df = do_transforms(transforms[0], df)

#   input_filters = parameters['inputFilters']
#   input_parameters = parameters['features']
#   df1 = pd.DataFrame(index=df.index)
#   idx = 0
#   for col in input_parameters:
#     if algorithmType != 0 and col == parameters['trainLabel']:
#       df1[col] = df[col]
#       idx = idx +1
#       continue
#     if input_filters[idx]:
#       df1[col] = df[col]
#     idx = idx +1
  
#   df_train = None
#   df_test = None
#   df_train_index = None
#   df_test_index = None

#   train_label = parameters['trainLabel']

#   if 'disableSplit' in parameters and parameters['disableSplit']:
#     return [[df1, None]]

#   if 'kFold' in parameters and parameters['kFold'] != 0:
#     kf = KFold(n_splits= int(parameters['kFold']))
#     kf.get_n_splits(df1)
#     res = []
#     for train_index, test_index in kf.split(df1):
#       df_train_index = train_index[train_index<df1.shape[0]-train_shift]
#       df_train_index.sort()
#       df_test_index = test_index[test_index<df1.shape[0]-train_shift]
#       df_test_index.sort()
#       df_train = df1.loc[df_train_index, :]
#       df_test = df1.loc[df_test_index, :]
      
#       df_train_labels = df1.loc[df_train_index+train_shift, :]
#       df_test_labels = df1.loc[df_test_index+train_shift, :]

#       if algorithmType == 0 or algorithmType == 5:
#         res.append([df_train, df_test])
#         continue
      
#       if train_label == '':
#         return [[df_train, df_test, df_train_labels, df_test_labels]]
#       df2 = pd.DataFrame(index=df.index)
#       if train_label == 'triple_barrier':
#         triple_option = parameters['tripleOptions']
#         df2[train_label] = triple_barrier(df, triple_option['up'], triple_option['down'], triple_option['maxHold'])
#         df2[train_label] = df2[train_label].astype('category', copy=False)
#       else:
#         df2[train_label] = df[train_label]
      
#       df_train_labels = df2.loc[df_train_index+train_shift, :]
#       df_test_labels = df2.loc[df_test_index+train_shift, :]

#       res.append([df_train, df_test, df_train_labels, df_test_labels])
#     return res
#   elif random_select:
#     df_train = df1.sample(n=train_count, random_state=1)
#     df_train = df_train.sort_index(axis=0)
#     df_train_index = df_train.index - train_shift
#     df_train_index = df_train_index[df_train_index > 0]
#     df_train = df1.loc[df_train_index, :]

#     df_test_index = df1.index.difference(df_train.index) - train_shift
#     df_test_index = df_test_index[df_test_index > 0]
#     df_test = df1.loc[df_test_index, :]
#   else:
#     df_train = df1[df.index <= train_count]
#     df_train_index = df_train.index - train_shift
#     df_train_index = df_train_index[df_train_index > 0]
#     df_train = df1.loc[df_train_index, :]

#     df_test = df1[df.index > train_count]
#     df_test_index = df_test.index - train_shift
#     df_test_index = df_test_index[df_test_index > 0]
#     df_test = df1.loc[df_test_index, :]
  
#   df_train_labels = df1.loc[df_train_index+train_shift, :]
#   df_test_labels = df1.loc[df_test_index+train_shift, :]

#   if algorithmType == 0 or algorithmType == 5:
#     return [[df_train, df_test]]

#   if train_label == '':
#     return [[df_train, df_test, df_train_labels, df_test_labels]]
#   df2 = pd.DataFrame(index=df.index)
#   if train_label == 'triple_barrier':
#     triple_option = parameters['tripleOptions']
#     df2[train_label] = triple_barrier(df, triple_option['up'], triple_option['down'], triple_option['maxHold'])
#   else:
#     df2[train_label] = df[train_label]
  
#   df_train_labels = df2.loc[df_train_index+train_shift, :]
#   df_test_labels = df2.loc[df_test_index+train_shift, :]

#   return [[df_train, df_test, df_train_labels, df_test_labels]]


def kmean_clustering(input_file, transforms, parameters, optimize):
  df0 = pd.DataFrame(index=input_file.index)
  if 'Open' in input_file:
    df0['Ret'] = input_file.Open.shift(-2, fill_value=0) - input_file.Open.shift(-1, fill_value=0)
  
  train_data_set = prepare_train_test(input_file, transforms, parameters)
  res_data_set = []
  for df_train, df_test in train_data_set:
    noSplit = False
    if df_test is None:
      df_test =df_train
      noSplit = True
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
    if not noSplit:
      df_test['Tar'] = k_means.predict(df_test)
    graph = []

    if 'Open' in input_file:
      df0_test_ = df0_test['Ret'][df0_test.index<df0.shape[0]-2]
      graph.extend([np.cumsum(np.insert(df0_test_.loc[df_test['Tar'] == c].to_numpy(), 0, 0)) for c in range(0, n_clusters)])
    # graph.extend([np.insert(df0_test_.loc[df_test['Tar'] == c].to_numpy(), 0, 0) for c in range(0, n_clusters)])
    # graph = [df_train[col].to_numpy() for col in df_test]
    # graph = [df_test.loc[df_test['Tar'] == c].to_numpy() for c in range(0, n_clusters)]
    if 'Open' in input_file:
      metrics = [[df0_train['Ret'][df0_train.index<df0.shape[0]-2].loc[df_train['Tar'] == c].sum(), df0_test_.loc[df_test['Tar'] == c].sum()] for c in range(0, n_clusters) ]
    else:
      metrics = []
    
    features = []
    if df_test.shape[1] == 3:
      for c in range(0, n_clusters):
        df_test_1 = df_test[df_test['Tar'] == c]
        features.append([df_test_1.values[:, 0], df_test_1.values[:, 1]])
    
    if optimize:
      res_data_set.append([np.array([X, Y]).T, {'n_clusters': n_clusters, 'init': init[0], 'random_state': random_state[0]}, features])
    else:
      res_data_set.append([graph, metrics, features])
  return res_data_set


def get_decision_boundaries(classifier, X_set, y_set, num_points_to_plot):
  X1, X2 = np.meshgrid(
    np.linspace(start=X_set[:, 0].min(), stop=X_set[:, 0].max(), num=num_points_to_plot),
    np.linspace(start=X_set[:, 1].min(), stop=X_set[:, 1].max(), num=num_points_to_plot))
  r = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
  r1 = r.reshape(X1.shape)

  contours = measure.find_contours(r1, 0)

  X_set, y_set = resample(X_set, y_set, n_samples=num_points_to_plot, stratify=y_set, replace=True)
  # contours = []
  # for i, j in enumerate(np.unique(y_set)):
  #   contours.append([X1[r1 == j], X2[r1 == j]])
  
  features = []
  for i, j in enumerate(np.unique(y_set)):
    features.append([X_set[y_set == j, 0], X_set[y_set == j, 1]])
  return contours, features


def upload_input_file(file):
  global input_file
  try:
    input_file = pd.read_csv (file)
    return True
  except:
    return False
