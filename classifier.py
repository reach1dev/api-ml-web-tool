from sklearn.datasets import make_classification


def over_sampling(X, y):
  from imblearn.over_sampling import RandomOverSampler
  ros = RandomOverSampler(random_state=0)
  X_resampled, y_resampled = ros.fit_resample(X, y)
  from collections import Counter
  print(sorted(Counter(y_resampled).items()))
  return X_resampled, y_resampled

def test_over_sample():
  X, y = make_classification(
    n_samples=5000, n_features=2, n_informative=2,
    n_redundant=0, n_repeated=0, n_classes=3,
    n_clusters_per_class=1,
    weights=[0.01, 0.05, 0.94],
    class_sep=0.8, random_state=0)
  X_resampled, y_resampled = over_sampling(X, y)

  from sklearn.svm import LinearSVC
  clf = LinearSVC()
  clf.fit(X_resampled, y_resampled)


if __name__ == "__main__":
  test_over_sample()
