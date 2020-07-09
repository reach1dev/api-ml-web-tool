import pandas as pd
import numpy as np


def do_transforms(transform, df):
  Y = pd.DataFrame(index=df.index)
  if 'children' not in transform:
    return df
  for child in transform['children']:
    X = df.copy()
    X1 = transform_data(X, child, transform['id'])
    X2 =  do_transforms(child, X1)
    if X2 is not None:
      common_cols = list(set.intersection(*(set(x.columns) for x in [Y, X2])))
      new_cols = []
      for c in X2:
        if c not in common_cols:
          new_cols.append(c)
      Y = pd.concat([Y, X2[new_cols]], join='inner', axis=1)
  return Y


def transform_data(df, transform, parentId):
  tool = transform['tool']['id']
  inputs = transform['inputParameters']
  outputs = transform['outputParameters']
  params = transform['parameters']

  if tool == 114:
    df[params['name']] = df.eval(params['expression'])
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)
  
  rolling = params['rolling'] if 'rolling' in params else None
  for col in inputs:    
    do_fill_na = True
    if col in outputs:
      col_id = outputs[col]
      if tool == 101:
        df[col_id] = normalize_dataframe(df[col], rolling, params['min'], params['max'])
      elif tool == 102:
        df[col_id] = standard_dataframe(df[col], rolling)
      elif tool == 103:
        df[col_id] = fisher_transform(df[col])
      elif tool == 104:
        df[col_id] = subtract_median(df[col], rolling)
      elif tool == 105:
        df[col_id] = subtract_mean(df[col], rolling)
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
        df[col_id] = rolling_mean(df[col], rolling)
  if do_fill_na:
    return df.fillna(0)
  return df


def normalize_dataframe(df, length=20, min=0, max=1):
  df_cr = df.rolling(length) if length is not None else df
  return (((df - df_cr.min()) / (df_cr.max() - df_cr.min())) * (max-min) + min).fillna(0)


def standard_dataframe(df, length=20):
  df_cr = df.rolling(length) if length is not None else df
  df_res = (df - df_cr.mean()) / df_cr.std()
  return df_res if length is None else df_res.fillna(0)


def fisher_transform(df):
  # [np.log((1.0+v)/(1-v)) * .5 for v in df[col]]
  return [0 if v==1 else np.log((1.0+v)/(1-v)) * .5 for v in df]


def rolling_mean(df, length=20):
  df_cr = df.rolling(length) if length is not None else df
  return df_cr.mean()


def subtract_mean(df, length=20):
  df_cr = df.rolling(length) if length is not None else df
  return df-df_cr.mean()


def subtract_median(df, length=20):
  df_cr = df.rolling(length) if length is not None else df
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
