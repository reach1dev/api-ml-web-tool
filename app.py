from flask import Flask
from flask import request
from io import StringIO
import pandas as pd
import numpy as np
import json
import sys
from flask_cors import CORS, cross_origin
from sklearn import preprocessing

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

input_data = []


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def transform_data(input_data, transform):
    if transform['tool']['id'] == 101:
        inputFilters = transform['inputFilters']
        if inputFilters is not None:
            for i in range(0, len(inputFilters)):
                if i < 2:
                    inputFilters[i] = False
            selections = np.array(inputFilters)
        else:
            selections = np.array([False, False, True, True, True, True, True, True])
        x = input_data[:, selections].T
        parameters = transform['parameters']
        minmax_scale = preprocessing.MinMaxScaler(feature_range=(parameters['min'], parameters['max'])).fit(x.T)
        x=minmax_scale.transform(x.T).T
        output_data = np.concatenate((input_data[:, :2], x.T), axis=1)
        return output_data
    return input_data


@app.route('/get-transform-data', methods=['POST'])
@cross_origin()
def get_transform_data():
    global input_data
    transforms = request.json['transforms']
    output_data = None
    last_input_data = None
    for transform in transforms:
        print(transform['id'])
        if transform['id'] == 1000:
            output_data = input_data
        else:
            last_input_data = output_data
            output_data = transform_data(output_data, transform)
    return json.dumps([last_input_data, output_data], default=default)


@app.route('/upload-input-data', methods=['POST'])
@cross_origin()
def upload_input_data():
    global input_data
    file = request.files['file']
    try:
        input_file = pd.read_csv (file)
        input_data = input_file.to_numpy()
    except:
        e = sys.exc_info()[0]
        print (e)
        return 'invalid file or data', 400
    return '', 200