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
import os, time, sys
from datetime import datetime
import threading


app = Flask(__name__)
cors = CORS(app)


INPUT_FILE_LIMIT = 2


def file_name(file_id):
    return 'tmp/' + file_id + '.csv'

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.route('/get-transform-data/<file_id>', methods=['POST'])
def get_transform_data(file_id):
    df = pd.read_csv(file_name(file_id))

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
    input_file = pd.read_csv(file_name(file_id))

    transforms = request.json['transforms']
    parameters = request.json['parameters']
    res_file_id = str(uuid.uuid4())
    def run_job(res_file_id):
        [graph, metrics] = engine.train_and_test(input_file, transforms, parameters)
        with open('tmp/' + res_file_id + '.dat', 'wb') as f:
            np.save(f, [graph, metrics])

    thread = threading.Thread(target=run_job, args=[res_file_id])
    thread.start()
    return json.dumps({'res_file_id': res_file_id}), 200


@app.route('/get-train-result/<res_file_id>', methods=['POST'])
def get_train_result(res_file_id):
    file_path = 'tmp/' + res_file_id + '.dat'
    failed_file_path = 'tmp/' + res_file_id + '.failed'
    if os.path.exists(failed_file_path):
        os.remove(failed_file_path)
        return '', 203
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            [graph, metrics] = np.load(f, allow_pickle=True)
            os.remove(file_path)
            return json.dumps([graph, metrics], default=default)
    return '', 204


@app.route('/upload-input-data', methods=['POST'])
def upload_input_data():
    file = request.files['file']
    try:
        file_id = str(uuid.uuid4())
        df = pd.read_csv (file, index_col='Date')
        df.to_csv(file_name(file_id))
        path, dirs, files = next(os.walk("tmp"))
        files = [ fi for fi in files if not fi.endswith(".csv") ]
        file_count = len(files)
        if file_count > INPUT_FILE_LIMIT:
            for f in os.listdir("tmp"):
                if f.endswith(".csv") and os.stat(os.path.join(path,f)).st_mtime < datetime.now().timestamp() - 60*60:
                    os.remove(os.path.join('tmp/', f))

        return json.dumps({'file_id': file_id, 'columns': df.columns.values}, default=default), 200
    except Exception as e:
        print(e)
        return '', 400

# app.run()
