import pandas as pd
from transformations import do_transforms
from sklearn.model_selection import KFold


def transform_data(X, transforms, trained_params):
  for transform in transforms:
    for tr in transforms:
      if tr['id'] != transform['id'] and 'parentId' in transform and tr['id'] == transform['parentId']:
        if 'children' not in tr:
            tr['children'] = [transform]
        else:
            tr['children'].append(transform)
  
  return do_transforms(transforms[0], X, trained_params)


def filter_target(X, type, input_filters, input_parameters, train_label):
  Y = pd.DataFrame(index=X.index)
  idx = 0
  for col in input_parameters:
    if type != 0 and col == train_label:
      Y[col] = X[col]
      idx = idx +1
      continue
    if input_filters[idx]:
      Y[col] = X[col]
    idx = idx +1
  return Y


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


def train_test_split(X, type, k_fold, random_select, n_test_shift, n_train_size):
  if k_fold:
    kf = KFold(n_splits= k_fold)
    kf.get_n_splits(X)
    res = []
    for train_index, test_index in kf.split(X):
      X_train_index = train_index[train_index<X.shape[0]-n_test_shift]
      X_train_index.sort()
      X_test_index = test_index[test_index<X.shape[0]-n_test_shift]
      X_test_index.sort()
      X_train = X.loc[X_train_index, :]
      X_test = X.loc[X_test_index, :]

      res.append([X_train, X_test])
    return res
  
  if random_select:
    X_train = X.sample(n=n_train_size, random_state=1)
    X_train = X_train.sort_index(axis=0)
    X_train_index = X_train.index - n_test_shift
    X_train_index = X_train_index[X_train_index >= 0]
    X_train = X.loc[X_train_index, :]

    X_test_index = X.index.difference(X_train.index) - n_test_shift
    X_test_index = X_test_index[X_test_index >= 0]
    X_test = X.loc[X_test_index, :]
  else:
    X_train = X[X.index <= n_train_size]
    X_train_index = X_train.index - n_test_shift
    X_train_index = X_train_index[X_train_index >= 0]
    X_train = X.loc[X_train_index, :]

    X_test = X[X.index > n_train_size]
    X_test_index = X_test.index - n_test_shift
    X_test_index = X_test_index[X_test_index > n_train_size]
    X_test = X.loc[X_test_index, :]

  return [[X_train, X_test]]


def prepare_train_test(X, transforms, parameters):
  n_samples = X.shape[0]
  n_train_size = int(n_samples * 0.8)
  if 'trainSampleCount' in parameters:
    n_train_size = parameters['trainSampleCount']
  
  if 'type' in parameters:
    type = parameters['type']
  else:
    return []

  X0 = X.copy()
  if 'Date' in X:
    del X0['Date']
  if 'Time' in X:
    del X0['Time']

  # X1 = X0 # transform_data(X0, transforms)

  if type == 0 or type == 5:
    train_label = ''
  else:
    if 'trainLabel' in parameters:
      train_label = parameters['trainLabel']
    else:
      return []
  
  if 'inputFilters' in parameters:
    input_filters = parameters['inputFilters']
  else:
    return []
  if 'features' in parameters:
    input_parameters = parameters['features']
  else:
    return []
  
  if 'disableSplit' in parameters and parameters['disableSplit']:
    X2 = transform_data(X0, transforms, {})
    return [[X2, None]]
  
  k_fold = None
  if 'kFold' in parameters and parameters['kFold'] != 0:
    k_fold =int(parameters['kFold'])
  random_select = False
  if 'random_select' in parameters:
    random_select = parameters['randomSelect']
  n_test_shift = 0
  if 'testShift' in parameters:
    n_test_shift = parameters['testShift']
    if train_label == 'triple_barrier':
      n_test_shift = 0
  
  resampling = None
  if 'resampling' in parameters:
    resampling = parameters['resampling']
  
  # X3, Y1 = resample(X2, Y, resampling)
  result = train_test_split(X0, type, k_fold, random_select, n_test_shift, n_train_size)
  result_new = []
  triple_option = parameters['tripleOptions']

  if type != 0 and type != 5:
    Y = pd.DataFrame(index=X0.index)
    if train_label == 'triple_barrier':
      Y[train_label] = triple_barrier(X, triple_option['up'], triple_option['down'], triple_option['maxHold'])
    else:
      X2, _ = transform_data(X0, transforms, {})
      Y[train_label] = X2[train_label]

  for res in result:
    X_train, X_test = res
    
    X_train, trained_params = transform_data(X_train, transforms, {})
    X_test, _ = transform_data(X_test, transforms, trained_params)
    
    if type == 0 or type == 5:
      if type != 0:
        X_train = filter_target(X_train, type, input_filters, input_parameters, train_label)
        X_test = filter_target(X_test, type, input_filters, input_parameters, train_label)
      result_new.append([X_train, X_test])
      continue

    if train_label != 'triple_barrier':
      Y.loc[X_train.index, train_label] = X_train.loc[X_train.index, train_label]
      Y.loc[X_test.index, train_label] = X_test.loc[X_test.index, train_label]
    Y_train = Y.loc[X_train.index+n_test_shift, :]
    Y_test = Y.loc[X_test.index+n_test_shift, :]

    X_train = filter_target(X_train, type, input_filters, input_parameters, train_label)
    X_test = filter_target(X_test, type, input_filters, input_parameters, train_label)

    X_train, Y_train = resample(X_train, Y_train, resampling)
    result_new.append([X_train, X_test, Y_train, Y_test])
  return result_new



def resample(X, Y, resampling):
  X_resampled, y_resampled = X, Y
  if resampling == 'oversampling':
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, Y)
  if resampling == 'undersampling':
    from imblearn.under_sampling import ClusterCentroids
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, Y)
  if resampling == 'smote':
    from imblearn.over_sampling import BorderlineSMOTE
    X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, Y)
  return X_resampled.fillna(0), y_resampled.fillna(0)
