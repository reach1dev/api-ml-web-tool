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
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
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


def get_metrics(y_true, y_pred, is_regression, algorithmType):
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
  for df_train, df_test, df_train_labels, df_test_labels, _ in train_data_set:
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
  for df_train, _, df_train_labels, _, _ in train_data_set:
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
      params = ['max_depth', 'random_state', 'criterion']
      if parameters['regression']:
        classifier = DecisionTreeRegressor()
      else:
        classifier = DecisionTreeClassifier()
    elif algorithmType == 8:
      main_param = 'max_depth'
      params = ['max_depth', 'random_state', 'criterion']
      if parameters['regression']:
        classifier = RandomForestRegressor
      else:
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
  
  for X_train, X_test, y_train, y_test, trained_params in train_data_set:
    label = parameters['trainLabel']
    
    if algorithmType == 1:
      classifier = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'], p=parameters['P'], metric=parameters['metric'])
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
        classifier = DecisionTreeClassifier(
          max_depth=parameters.get('max_depth', 2), 
          random_state=parameters.get('random_state', 0), 
          criterion=parameters.get('criterion', 'gini')
        )
      else:
        classifier = DecisionTreeRegressor(
          max_depth=parameters.get('max_depth', 2), 
          random_state=parameters.get('random_state', 0), 
          criterion=parameters.get('criterion', 'mse')
        )
    elif algorithmType == 8:
      if not parameters.get('regression', False):
        classifier = RandomForestClassifier(
          max_depth=parameters.get('max_depth', 2), 
          random_state=parameters.get('random_state', 0), 
          n_estimators=parameters.get('n_estimators', 100), 
          criterion=parameters.get('criterion', 'gini')
        )
      else:
        classifier = RandomForestRegressor(
          max_depth=parameters.get('max_depth', 2), 
          random_state=parameters.get('random_state', 0), 
          n_estimators=parameters.get('n_estimators', 100), 
          criterion=parameters.get('criterion', 'mse')
        )
    elif algorithmType == 9:
      layers = parameters.get('hidden_layer_sizes', '5,2').split(',')
      hidden_layers = []
      for layer in layers:
        hidden_layers.append(int(layer))
      batch_size = parameters.get('batch_size', 'auto')
      batch_size = int(batch_size) if batch_size != 'auto' else 'auto'
      if not parameters.get('regression', False):
        classifier = MLPClassifier(
          hidden_layer_sizes=hidden_layers,
          solver=parameters.get('solver', 'sgd'),
          alpha=parameters.get('alpha', 0.00001),
          random_state=parameters.get('random_state', 0),
          learning_rate_init=parameters.get('learning_rate_init', 0.001),
          learning_rate=parameters.get('learning_rate', 'constant'),
          batch_size=batch_size,
          max_iter=500
        )
      else:
        classifier = MLPRegressor(
          hidden_layer_sizes=hidden_layers,
          solver=parameters.get('solver', 'sgd'),
          alpha=parameters.get('alpha', 0.00001),
          random_state=parameters.get('random_state', 0),
          learning_rate_init=parameters.get('learning_rate_init', 0.001),
          learning_rate=parameters.get('learning_rate', 'constant'),
          batch_size=batch_size,
          max_iter=500
        )
      
    y_train = y_train[label]
    y_test = y_test[label]
    is_regression = algorithmType == 2 or (algorithmType==4 and parameters.get('useSVR', False)) or (parameters.get('regression', False))

    if label != 'triple_barrier' and not is_regression:
      y_train = y_train.astype('int').astype('category')
      y_test = y_test.fillna(0).astype('int').astype('category')
    
    for idx, col in enumerate(parameters['features']):
      if col == label and parameters['inputFilters'][idx] == False:
        del X_train[label]
        del X_test[label]
        break
    
    classifier.fit(X_train, y_train)

    # if algorithmType == 3 or algorithmType == 4:
    #   df_train = df_train.astype('int').astype('category')
    #   df_test = df_test.astype('int').astype('category')
    p_train = classifier.predict(X_train)
    p_test = classifier.predict(X_test)
    
    test_shift = parameters['testShift'] if parameters['trainLabel'] != 'triple_barrier' else 0
    y_test1 = y_test.dropna()
    df_test_score, df_test_cm = get_metrics(y_test1, p_test[:len(y_test1)], is_regression, algorithmType)
    df_train_score, _ = get_metrics(y_train, p_train, is_regression, algorithmType)
    
    date_index = input_file.loc[y_test.index, input_file.columns[0]]
    date_index = date_index.dropna()
    res = [np.array(date_index) ]
    
    y_test = y_test.fillna(0).to_numpy()
    # p_test = p_test
    res.append(y_test)
    res.append(p_test)
    
    contours, features = [[], []]
    if not is_regression and X_test.shape[1] == 2:
      X_train = pd.DataFrame(index=X_train.index)
      for col in input_file:
        if col == 'No' or X_train.shape[1] >= 2:
          continue
        X_train[col] = input_file.loc[X_train.index, col]
      
      X_test = pd.DataFrame(index=X_test.index)
      for col in input_file:
        if col == 'No' or X_test.shape[1] >= 2:
          continue
        X_test[col] = input_file.loc[X_test.index, col]
      contours_train, features_train = get_decision_boundaries(classifier, X_train, y_train, 200, transforms, trained_params, algorithmType, parameters)
      features_test = get_features(X_test, y_test)
      features = np.array([features_train, features_test])
      contours = contours_train
    if is_regression and X_test.shape[1] == 1:
      features = np.array([[X_train[X_train.columns[0]].values, y_train.values], [X_test[X_test.columns[0]].values, y_test]])
      contours = np.array([[X_train[X_train.columns[0]].values, p_train], [X_test[X_test.columns[0]].values, p_test]])

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


def get_features(df_train, y_set):
  X_set = df_train.values
  features = []
  for i, j in enumerate(np.unique(y_set)):
    features.append([X_set[y_set == j, 0], X_set[y_set == j, 1]])
  return features

def get_decision_boundaries(classifier, df_train, y_set, num_points_to_plot, transforms, trained_params, type, extra = {}):
  X_set = df_train.values
  dx1 = (X_set[:, 0].max() - X_set[:, 0].min()) * 0.05
  dx2 = (X_set[:, 1].max() - X_set[:, 1].min()) * 0.05
  X1, X2 = np.meshgrid(
    np.linspace(start=X_set[:, 0].min() - dx1, stop=X_set[:, 0].max() + dx1, num=num_points_to_plot),
    np.linspace(start=X_set[:, 1].min() - dx2, stop=X_set[:, 1].max() + dx2, num=num_points_to_plot))
  
  X_train = pd.DataFrame()
  X_train[df_train.columns[0]] = X1.ravel()
  X_train[df_train.columns[1]] = X2.ravel()

  from preprocessor import transform_data
  from preprocessor import filter_target
  X_train, _ = transform_data(X_train, transforms, trained_params)
  X_train = filter_target(X_train, type, extra['inputFilters'], extra['features'], '')

  r = classifier.predict(X_train.replace([np.inf, -np.inf], np.nan).fillna(0))
  r1 = r.reshape(X1.shape)

  contours = measure.find_contours(r1, 0)
  f_contours = []
  for contour in contours:
    xa1 = []
    xa2 = []
    for pt in contour:
      xa1.append(X1[int(pt[0])][int(pt[1])])
      xa2.append(X2[int(pt[0])][int(pt[1])])
    f_contours.append([xa1, xa2])

  # X_set, y_set = resample(X_set, y_set, n_samples=num_points_to_plot, stratify=y_set, replace=True)
  
  features = []
  for i, j in enumerate(np.unique(y_set)):
    features.append([X_set[y_set == j, 0], X_set[y_set == j, 1]])
  return f_contours, features


def upload_input_file(file):
  global input_file
  try:
    input_file = pd.read_csv (file)
    return True
  except:
    return False
