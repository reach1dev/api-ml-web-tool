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
from constants import get_x_unit
from transformations import transform_data
import redis


app = Flask(__name__)
cors = CORS(app)


INPUT_FILE_LIMIT = 2


def file_name(file_id: str):
    return 'tmp/' + file_id + '.csv'


def get_input_file(file_id: str):
    if 'data_' in file_id:
        rd = redis.from_url(os.environ.get("REDIS_URL"))
        df = pd.read_msgpack(rd.get(file_id.replace('data_', '')))
    else:
        file_path = file_name(file_id)
        df = pd.read_csv(file_path)
    return df


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.route('/get-transform-data/<file_id>', methods=['POST'])
def get_transform_data(file_id):
    df = get_input_file(file_id)

    transforms = request.json['transforms']
    output_data = None
    last_transform = transforms[0]
    N = get_x_unit(df.shape[0])
    try:
        for transform in transforms:
            if transform['id'] == 1000:
                output_data = df.copy()
            else:
                output_data, _ = transform_data(output_data, transform, last_transform['id'], {})
        
        columns = []
        if output_data is not None:
            output_data = output_data.dropna()
            columns = pd.DataFrame(data=[output_data.columns.values,output_data.min(),output_data.max()]).to_numpy()
            output_data = output_data.to_numpy()
            # sel = [i%N==0 for i in range(len(output_data))]
            # output_data = output_data[sel]
        return json.dumps({'columns': columns, 'data': output_data}, default=default)
    except Exception as e:
        traceback.print_exc()
        return repr(e), 203


@app.route('/train-and-test/<file_id>', methods=['POST'])
def train_and_test(file_id):
    return inter_train(file_id)


@app.route('/save-model', methods=['POST'])
def save_model_to_db():
    transforms = request.json['transforms']
    parameters = request.json['parameters']
    model_name = request.json['modelName']
    from database import save_model
    return { 'success': save_model(1, 1, model_name, transforms, parameters) }


@app.route('/update-model/<model_id>', methods=['PUT'])
def update_model_to_db(model_id):
    transforms = request.json['transforms']
    parameters = request.json['parameters']
    from database import update_model
    return { 'success': update_model(model_id, transforms, parameters) }


@app.route('/list-model', methods=['GET'])
def list_model_from_db():
    from database import load_models
    return load_models(1)


@app.route('/optimize/<file_id>', methods=['POST'])
def optimize(file_id):
    return inter_train(file_id, optimize=True)


def inter_train(file_id, optimize = False):
    input_file = get_input_file(file_id)
    return inter_train_with_file(input_file, optimize, request.json['transforms'], request.json['parameters'])


def inter_train_with_file(input_file, optimize, transforms, parameters):
    res_file_id = str(uuid.uuid4())
    def run_job(res_file_id):
        try:
            res_data_set = engine.train_and_test(input_file, transforms, parameters, optimize=optimize)
            # from trainer import train_and_test
            # res_data_set = train_and_test(input_file, transforms, parameters)
            with open('tmp/' + res_file_id + '.dat', 'wb') as f:
                header = np.array([len(res_data_set)])
                np.save(f, np.array(header))
                for row in res_data_set:
                    graph = None
                    metrics = None
                    cm = []
                    contours = []
                    features = []
                    if len(row) == 5:
                        [graph, metrics, cm, contours, features] = row
                    elif len(row) == 3:
                        [graph, metrics, features] = row
                    else:
                        [graph, metrics] = row
                    np.save(f, graph)
                    np.save(f, metrics)
                    np.save(f, cm)
                    np.save(f, contours)
                    np.save(f, features)
                f.close()
        except Exception as e:
            print(e)
            with open('tmp/' + res_file_id + '.err', 'w') as f:
                f.writelines([repr(e)])
                f.close()

    thread = threading.Thread(target=run_job, args=[res_file_id])
    thread.start()
    return json.dumps({'res_file_id': res_file_id}), 200

@app.route('/get-train-result/<res_file_id>', methods=['POST'])
def get_train_result(res_file_id):
    file_path = 'tmp/' + res_file_id + '.dat'
    err_path = 'tmp/' + res_file_id + '.err'
    if os.path.exists(err_path):
        with open(err_path, 'r') as f:
            err = f.readline()
            f.close()
            os.remove(err_path)
            return json.dumps({'err': err}, default=default), 203
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            header = np.load(f, allow_pickle=True)
            res_count = header[0]
            res_data_set = []
            for k in range(res_count):
                graph = np.load(f, allow_pickle=True)
                metrics = np.load(f, allow_pickle=True)
                cm = np.load(f, allow_pickle=True)
                contours = np.load(f, allow_pickle=True)
                features = np.load(f, allow_pickle=True)
                res_data_set.append([graph, metrics, cm, contours, features])
            f.close()
            os.remove(file_path)
            return json.dumps(res_data_set, default=default)
    return '', 204


@app.route('/upload-input-data/<has_index>', methods=['POST'])
def upload_input_data(has_index):
    file = request.files['file']
    try:
        file_id = str(uuid.uuid4())
        if has_index == '1':
            df = pd.read_csv (file, index_col=0)
        else:
            df = pd.read_csv (file, index_col=False)
            df1 = pd.DataFrame()
            df1['No'] = np.arange(len(df))
            df.index = df1['No']
        df.to_csv(file_name(file_id))
        path, dirs, files = next(os.walk("tmp"))
        files = [ fi for fi in files if not fi.endswith(".csv") ]
        file_count = len(files)
        if file_count > INPUT_FILE_LIMIT:
            for f in os.listdir("tmp"):
                if f.endswith(".csv") and os.stat(os.path.join(path,f)).st_mtime < datetime.now().timestamp() - 60*60:
                    os.remove(os.path.join('tmp/', f))

        return json.dumps({'file_id': file_id, 'index': df.index.name, 'columns': df.columns.values, 'sample_count': len(df)}, default=default), 200
    except Exception as e:
        print(e)
        return '', 400


@app.route('/select-input-data/<file_id>', methods=['POST'])
def select_input_data(file_id):
    try:
        df = get_input_file(file_id)

        return json.dumps({'file_id': file_id, 'index': 'Date', 'columns': df.columns.values[1:], 'sample_count': len(df)}, default=default), 200
    except Exception as e:
        print(e)
        return '', 400


@app.route('/test-autoupdate', methods=['GET'])
def test_autoupdate():
    from autoupdate import autoupdate
    return autoupdate(), 200

# app.run()
