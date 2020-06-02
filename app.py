from flask import Flask
from flask import request
from io import StringIO
import pandas as pd
import numpy as np
import json
import sys
from flask_cors import CORS, cross_origin
from sklearn import preprocessing
from sklearn.cluster import KMeans
import engine

app = Flask(__name__)
cors = CORS(app)


input_data = []


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.route('/get-transform-data', methods=['POST'])
def get_transform_data():
    df = engine.copy_dataframe()
    transforms = request.json['transforms']
    output_data = None
    last_input_data = None
    for transform in transforms:
        if transform['id'] == 1000:
            output_data = df
        elif transform['tool']['id'] == 101:
            last_input_data = output_data.copy()
            output_data = engine.normalize_dataframe(output_data, transform['outputParameters'], transform['parameters']['rolling'])
    
    if last_input_data is not None:
        last_input_data = last_input_data.to_numpy()
        sel = [i%50==0 for i in range(len(last_input_data))]
        last_input_data = last_input_data[sel]
    if output_data is not None:
        output_data = output_data.to_numpy()
        sel = [i%50==0 for i in range(len(output_data))]
        output_data = output_data[sel]
    return json.dumps([last_input_data, output_data], default=default)


@app.route('/train-and-test', methods=['POST'])
def train_and_test():
    transforms = request.json['transforms']
    parameters = request.json['parameters']
    
    [graph, metrics] = engine.train(transforms, parameters)
    return json.dumps([graph, metrics], default=default)


@app.route('/upload-input-data', methods=['POST'])
def upload_input_data():
    global input_data
    file = request.files['file']
    if engine.upload_input_file(file):
        return '', 200
    return '', 400
