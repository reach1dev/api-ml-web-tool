import pandas as pd
import numpy as np



def do_transforms(transform, df, trained_params):
  Y = pd.DataFrame(index=df.index)
  if 'children' not in transform:
    return df, trained_params
  for child in transform['children']:
    X = df.copy()
    X1, trained_params = transform_data(X, child, transform['id'], trained_params)
    X2, trained_params = do_transforms(child, X1, trained_params)
    if X2 is not None:
      common_cols = list(set.intersection(*(set(x.columns) for x in [Y, X2])))
      new_cols = []
      for c in X2:
        if c not in common_cols:
          new_cols.append(c)
      Y = pd.concat([Y, X2[new_cols]], join='inner', axis=1)
  return Y.dropna(), trained_params


def transform_data(df, transform, parentId, trained_params):
  tool = transform['tool']['id']
  inputs = transform['inputParameters']
  outputs = transform['outputParameters']
  params = transform['parameters']

  if tool == 114:
    try:
      df[params['name']] = df.eval(params['expression'])
    except Exception as e:
      print(e)
      return df, trained_params
    return df.replace([np.inf, -np.inf], np.nan).fillna(0), trained_params
  
  rolling = params['rolling'] if 'rolling' in params else None
  tid = transform['id']
  if tid not in trained_params:
    trained_params[tid] = {}
  for col in inputs:    
    do_fill_na = True
    if col in outputs and col in df:
      col_id = outputs[col]
      if col_id not in trained_params[tid]:
        trained_params[tid][col_id] = {}
      if tool == 101:
        df[col_id], trained_params[tid][col_id] = normalize_dataframe(df[col], rolling, params['min'], params['max'],  trained_params[tid][col_id])
      elif tool == 102:
        df[col_id], trained_params[tid][col_id] = standard_dataframe(df[col], rolling, trained_params[tid][col_id])
      elif tool == 103:
        df[col_id] = fisher_transform(df[col])
      elif tool == 104:
        df[col_id], trained_params[tid][col_id] = subtract_median(df[col], rolling, trained_params[tid][col_id])
      elif tool == 105:
        df[col_id], trained_params[tid][col_id] = subtract_mean(df[col], rolling, trained_params[tid][col_id])
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
        df[col_id] = turn_percentile(df[col], rolling)
      elif tool == 113:
        df[col_id] = power_function(df[col], params['power'])
      elif tool == 115:
        df[col_id], trained_params[tid][col_id] = rolling_mean(df[col], rolling, trained_params[tid][col_id])
  if do_fill_na:
    return df.dropna(), trained_params
  return df, trained_params


def normalize_dataframe(df, length=20, min=0, max=1, rp={}):
  df_cr = df.rolling(length) if length is not None else df
  if length is None:
    if 'min' not in rp:
      rp['min'] = df.min()
    if 'max' not in rp:
      rp['max'] = df.max()
  df_cr_min = df_cr.min() if length is not None else rp['min']
  df_cr_max = df_cr.max() if length is not None else rp['max']
  return (((df - df_cr_min) / (df_cr_max - df_cr_min)) * (max-min) + min).fillna(0), rp


def standard_dataframe(df, length=20, rp={}):
  df_cr = df.rolling(length) if length is not None else df
  if length is None:
    if 'mean' not in rp:
      rp['mean'] = df.mean()
    if 'std' not in rp:
      rp['std'] = df.std()
  df_cr_mean = df_cr.mean() if length is not None else rp['mean']
  df_cr_std = df_cr.std() if length is not None else rp['std']
  df_res = (df - df_cr_mean) / df_cr_std
  return df_res if length is None else df_res.fillna(0), rp


def fisher_transform(df):
  # [np.log((1.0+v)/(1-v)) * .5 for v in df[col]]
  return [0 if v==1 else np.log((1.0+v)/(1-v)) * .5 for v in df]


def rolling_mean(df, length=20, rp={}):
  df_cr = df.rolling(length) if length is not None else df
  if length is None:
    if 'mean' not in rp:
      rp['mean'] = df.mean()
  return df_cr.mean() if length is not None else rp['mean'], rp


def subtract_mean(df, length=20, rp={}):
  df_cr = df.rolling(length) if length is not None else df
  if length is None:
    if 'mean' not in rp:
      rp['mean'] = df.mean()
  df_cr_mean = df_cr.mean() if length is not None else rp['mean']
  return df-df_cr_mean, rp


def subtract_median(df, length=20, rp={}):
  df_cr = df.rolling(length) if length is not None else df
  if length is None:
    if 'median' not in rp:
      rp['median'] = df.median()
  df_cr_median = df_cr.median() if length is not None else rp['median']
  return df-df_cr_median, rp


def first_diff(df, length=1):
  return df - df.shift(length)


def percent_return(df, length=1):
  return ((df - df.shift(length))/df.shift(length)) * 100.0


def log_return(df, length=1):
  # np.log(df['Close']/df['Close'].shift(x))
  return np.log(df / (df.shift(length))).replace(np.inf, 0).replace(-np.inf, 0)


def clip_dataframe(df, min, max, scale=1):
  if isinstance(min, str):
    min = float(min)
  if isinstance(max, str):
    max = float(max)
  if isinstance(scale, str):
    scale = float(scale)
  return df.clip(min, max) * scale


def turn_categorical(df):
  df1 = df.astype('int')
  df2 = df1 - df1.min()
  return df2.astype('category', copy=False)

def turn_ranking(df):
  return df.rank()


def turn_percentile(df, length=20):
  if length is None:
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
