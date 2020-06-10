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
import uuid
import datetime
import time
import traceback


app = Flask(__name__)
cors = CORS(app)


input_files = {}
INPUT_FILE_LIMIT = 10


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.route('/get-transform-data/<file_id>', methods=['POST'])
def get_transform_data(file_id):
    global input_files
    if file_id not in input_files:
        return '', 405
    df = input_files[file_id]['file']
    input_files[file_id]['timestamp'] = datetime.datetime.now()

    transforms = request.json['transforms']
    output_data = None
    last_transform = transforms[0]
    N = int(df.shape[0] / 100.0)
    try:
        for transform in transforms:
            if transform['id'] == 1000:
                output_data = df.copy()
            else:
                output_data = engine.transform_data(output_data, transform, last_transform['id'])
        
        columns = []
        if output_data is not None:
            output_data = output_data[output_data.index % N == 0]
            columns = pd.DataFrame(data=[output_data.columns.values,output_data.min(),output_data.max()]).to_numpy()
            output_data = output_data.to_numpy()
            # sel = [i%N==0 for i in range(len(output_data))]
            # output_data = output_data[sel]
        return json.dumps({'columns': columns, 'data': output_data}, default=default)
    except:
        traceback.print_exc()
        return '', 402


@app.route('/train-and-test/<file_id>', methods=['POST'])
def train_and_test(file_id):
    global input_files
    if file_id not in input_files:
        return '', 400
    input_file = input_files[file_id]['file']
    input_files[file_id]['timestamp'] = datetime.datetime.now()

    transforms = request.json['transforms']
    parameters = request.json['parameters']
    [graph, metrics] = engine.train_and_test(input_file, transforms, parameters)
    return json.dumps([graph, metrics], default=default)


@app.route('/upload-input-data', methods=['POST'])
def upload_input_data():
    global input_files
    file = request.files['file']
    try:
        file_id = str(uuid.uuid4())
        df = pd.read_csv (file)
        input_files[file_id] = {
            'file': df,
            'timestamp': datetime.datetime.now()
        }
        if len(input_files) > INPUT_FILE_LIMIT:
            old_file_id = None
            old_time = None
            for file_id in input_files.keys():
                t1 = time.mktime(datetime.datetime.now().timetuple())
                t2 = time.mktime(input_files[file_id]['timestamp'].timetuple())
                time_diff = t1-t2
                if old_time is None or time_diff > old_time:
                    old_time = time_diff
                    old_file_id = file_id
            if old_file_id is not None:
                del input_files[old_file_id]
        return json.dumps({'file_id': file_id, 'columns': df.columns.values}, default=default), 200
    except Exception as e:
        print(e)
        return '', 400

# app.run()
