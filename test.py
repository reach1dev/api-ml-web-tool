import engine
import json
import io
import matplotlib.pyplot as plt

with open('test_transforms.json') as transforms_file:
	engine.upload_input_file(r'test.csv')

	transforms = json.load(transforms_file)

	[graph, metrics] = engine.train(transforms, {'n_clusters': 3})
	print(metrics)

	plt.show()
	for g in graph:
		plt.plot(g)
	plt.show()
