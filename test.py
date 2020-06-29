# import engine
# import json
# import io
# import matplotlib.pyplot as plt

# with open('test_transforms.json') as transforms_file:
# 	engine.upload_input_file(r'test.csv')

# 	transforms = json.load(transforms_file)

# 	[graph, metrics] = engine.train(transforms, {'n_clusters': 3})
# 	print(metrics)

# 	plt.show()
# 	for g in graph:
# 		plt.plot(g)
# 	plt.show()

import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=5)
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
