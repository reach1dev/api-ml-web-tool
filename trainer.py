import numpy as np
import pandas as pd

def train_and_test(df, transforms, parameters):
  X = df.iloc[:,1:-1].values
  y = df.iloc[:, -1].values

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2)

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test) 

  from sklearn.neighbors import KNeighborsClassifier
  classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test)

  from sklearn.metrics import confusion_matrix, accuracy_score
  cm = confusion_matrix(y_test, y_pred)
  print(cm)
  print(accuracy_score(y_test, y_pred))

  res = [np.arange(X_test.shape[0]), y_test, y_pred]
  
  from engine import get_metrics
  train_score = get_metrics(y_test, y_pred, False, 1)
  test_score = get_metrics(y_test, y_pred, False, 1)
  score = np.array([train_score, test_score]).T

  contours, features = get_decision_boundaries(sc, classifier, X_train, y_train)
  return [[res, score, cm, contours, features]]


def get_decision_boundaries(sc, classifier, X_train, y_train):
  X_set, y_set = sc.inverse_transform(X_train), y_train
  
  X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1),
    np.arange(start=X_set[:, 1].min() - 10, stop=X_set[:, 1].max() + 10, step=500))

  X_set_new = np.array([X1.ravel(), X2.ravel()]).T
  X_set_new = sc.transform(X_set_new)
  r = classifier.predict(X_set_new)
  r1 = r.reshape(X1.shape)

  from skimage import measure
  contours = measure.find_contours(r1, 0)
  f_contours = []
  for contour in contours:
    xa1 = []
    xa2 = []
    for pt in contour:
      xa1.append(X1[int(pt[0])][int(pt[1])])
      xa2.append(X2[int(pt[0])][int(pt[1])])
    f_contours.append([xa1, xa2])
  
  features = []
  for i, j in enumerate(np.unique(y_set)):
    features.append([X_set[y_set == j, 0], X_set[y_set == j, 1]])
  
  return f_contours, features
  