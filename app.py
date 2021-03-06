from flask import Flask, redirect
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
from transformations import transform_data
import redis
import jwt
from flask_httpauth import HTTPTokenAuth, HTTPBasicAuth


app = Flask(__name__)
cors = CORS(app)
auth = HTTPTokenAuth(scheme='Bearer')

JWT_SECRET = 'secret'
JWT_ALGORITHM = 'HS256'


INPUT_FILE_LIMIT = 2


def file_name(file_id: str):
    return 'tmp/' + file_id + '.csv'


def get_input_file(file_id: str, refresh_token = None):
    if 'TSData_' in file_id:
        rd = redis.from_url(os.environ.get("REDIS_URL"))
        rd_file = rd.get(file_id)
        if rd_file is not None and rd_file != b'failed' and rd_file != b'waiting':
            df = pd.read_msgpack(rd_file)
        elif refresh_token is not None:
            file_params = file_id.split('_')
            if len(file_params) < 4:
                return None
            symbol = file_params[1]
            frequency = file_params[2]
            start_date = file_params[3]
            from tsapi import load_ts_prices
            from tsapi import get_access_token
            access_token = get_access_token(refresh_token)

            rd.set(file_id, 'waiting')
            def run_job(rd, file_id, access_token, symbol, frequency, start_date):            
                try:
                    df = load_ts_prices(access_token, symbol, frequency, start_date)
                    if df is None:
                        rd.set(file_id, 'failed')
                        return None
                    rd.set(file_id, df.to_msgpack(compress='zlib'))
                except Exception as e:
                    print(e)
                    rd.set(file_id, 'failed')


            thread = threading.Thread(target=run_job, args=[rd, file_id, access_token, symbol, frequency, start_date])
            thread.start()
            return False
        else:
            return None
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


@auth.verify_token
def verify_token(token):
    print(token)
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    t1 = datetime.strptime(payload['time'], '%Y-%m-%d %H:%M:%S')
    t2 = datetime.now()
    print("current time = " + t2.strftime('%Y-%m-%d %H:%M:%S'))
    print("token time = " + payload['time'])
    print("delta seconds = " + str((t2-t1).seconds))
    if (t2-t1).seconds < 12*3600:
        from database import find_user
        return find_user(payload['username'])
    return None


@app.route('/get-transform-data/<file_id>', methods=['POST'])
def get_transform_data(file_id):
    df = get_input_file(file_id)
    if df is None or df is False:
        return '', 204

    transforms = request.json['transforms']
    output_data = None
    last_transform = transforms[0]
    N = 1
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
@auth.login_required
def save_model_to_db():
    if auth.current_user() is None:
        return '', 401
    transforms = request.json['transforms']
    parameters = request.json['parameters']
    model_name = request.json['modelName']
    input_file_id = request.json['inputFileId']
    file_params = input_file_id.split("_")
    symbol = file_params[1]
    frequency = file_params[2]
    start_date = file_params[3] if len(file_params) > 3 else '01-01-2010'
    from database import save_model
    return { 'success': save_model(auth.current_user()['user_id'], model_name, symbol, frequency, start_date, transforms, parameters) }


@app.route('/update-model/<model_id>', methods=['PUT'])
@auth.login_required
def update_model_to_db(model_id):
    transforms = request.json['transforms']
    parameters = request.json['parameters']
    input_file_id = request.json['inputFileId']
    file_params = input_file_id.split("_")
    symbol = file_params[1]
    frequency = file_params[2]
    start_date = file_params[3] if len(file_params) > 3 else '01-01-2010'
    from database import update_model
    return { 'success': update_model(model_id, transforms, parameters, symbol, frequency, start_date) }


@app.route('/remove-model/<model_id>', methods=['DELETE'])
@auth.login_required
def remove_model_from_db(model_id):
    from database import remove_model
    return { 'success': remove_model(model_id) }


@app.route('/list-model', methods=['GET'])
@auth.login_required
def list_model_from_db():
    if auth.current_user() is None:
        return '', 401
    from database import load_models
    return load_models(auth.current_user()['user_id'])


@app.route('/optimize/<file_id>', methods=['POST'])
def optimize(file_id):
    return inter_train(file_id, optimize=True)


@app.route('/auth/login', methods=['POST'])
def auth_login():
    post_data = request.get_json()
    from database import check_user
    from database import clear_user_token
    user = check_user(post_data['username'], post_data['password'])
    if user is not None and (user['email'] is not None and user['email'] != ''):
        clear_user_token(post_data['username'])
        token = jwt.encode({
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'username': post_data['username']
        }, JWT_SECRET, algorithm=JWT_ALGORITHM).decode('utf-8')
        return {
            'success': True,
            'token': token,
            'fullName': user['fullName'],
            'email': user['email'],
            'webAlerts': user['webAlerts'],
            'emailAlerts': user['emailAlerts']
        }
    return {
        'success': False,
        'reason': 'no_account' if user is None else 'signup_required'
    }


@app.route('/auth/signup', methods=['POST'])
def auth_signup():
    post_data = request.get_json()
    from database import check_user
    user = check_user(post_data['username'], post_data['password'])
    print(user)
    if user is not None and (user['email'] is None or user['email'] == ''):
        from database import update_user
        if update_user(post_data['username'], post_data['email'], post_data['fullname']):
            token = jwt.encode({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'username': post_data['username']
            }, JWT_SECRET, algorithm=JWT_ALGORITHM).decode('utf-8')
            return {
                'success': True,
                'token': token
            }
        else:
            return {
                'success': False,
                'reason': 'creation_failed'
            }
    return {
        'success': False,
        'reason': 'no_account' if user is None else 'already_created'
    }


@app.route('/account/update', methods=['POST'])
@auth.login_required
def account_update():
    user = auth.current_user()
    post_data = request.get_json()
    from database import update_user
    if update_user(user['username'], post_data['email'], post_data['fullname']):
        return {
            'success': True
        }
    else:
        return {
            'success': False
        }


@app.route('/account/upload', methods=['POST'])
@auth.login_required
def account_upload():
    user = auth.current_user()
    if user['username'] == 'admin':
        file = request.files['file']
        df = pd.read_csv (file)
        users = []
        for _, row in df.iterrows():
            users.append([row['username'], row['password']])
        from database import upload_users
        res = upload_users(users)
        return {
            'success': res,
            'added_users': len(users) if res else 0
        }
    else:
        return {
            'success': False
        }


def inter_train(file_id, optimize = False):
    input_file = get_input_file(file_id)
    if input_file is None or input_file is False:
        return '', 404
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
            traceback.print_exc()
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


@app.route('/get-input-data/<file_id>', methods=['POST'])
@auth.login_required
def get_input_data(file_id):
    rd = redis.from_url(os.environ.get("REDIS_URL"))
    file = rd.get(file_id)
    if file is None or file == b'waiting':
        return {'file_id': file_id, 'status': 'waiting'}
    elif file == b'failed':
        return {'file_id': file_id, 'status': 'failed'}
    elif file is not None:
        df = pd.read_msgpack(file)
        return json.dumps({'file_id': file_id, 'status': 'success', 'index': 'Date', 'columns': df.columns.values[1:], 'sample_count': len(df)}, default=default), 200
    return {'file_id': file_id, 'status': 'failed'}


@app.route('/select-input-data/<file_id>', methods=['POST'])
@auth.login_required
def select_input_data(file_id):
    user = auth.current_user()
    if user['refresh_token'] is None or user['refresh_token'] == '':
        return '', 403
    df = get_input_file(file_id, refresh_token=user['refresh_token'])
    if df is None:
        return '', 403
    if df is False:
        return {'file_id': file_id, 'status': 'waiting'}
    return json.dumps({'file_id': file_id, 'status': 'success', 'index': 'Date', 'columns': df.columns.values[1:], 'sample_count': len(df)}, default=default), 200


@app.route('/test-autoupdate', methods=['GET'])
def test_autoupdate():
    from database import load_users
    result = load_users()
    if result['success']:
        for user in result['users']:
            if user['model_count'] > 0:
                # try:
                from autoupdate import autoupdate
                autoupdate(user['user_id'], user['username'], user['refresh_token'], user['email'], user['email_alert'])
                # except Exception as e:
                #     print(e)
    return '',200


@app.route('/account/alerts', methods=['PUT'])
@auth.login_required
def account_update_alerts_settings():
    username = auth.current_user()['username']
    post_data = request.get_json()
    from database import update_user_alerts
    if update_user_alerts(username, post_data['webAlerts'], post_data['emailAlerts']):
        return {
            'success': True
        }
    else:
        return {
            'success': False
        }


@app.route('/account/tsapi_callback/<token>', methods=['GET'])
def account_tsapi_callback(token):
    auth_code = request.args.get('code')
    user = verify_token(token)
    if user is None:
        return redirect("https://ml-web-tool.herokuapp.com/")
    username = user['username']
    from tsapi import get_token
    result = get_token(token, auth_code)
    if result is not None:
        from database import update_user_tsapi_tokens
        update_user_tsapi_tokens(username, result['access_token'], result['refresh_token'])
    return redirect("https://ml-web-tool.herokuapp.com/")


@app.route('/account/web-alert', methods=['GET'])
@auth.login_required
def get_web_alert():
    username = auth.current_user()['username']
    user_id = auth.current_user()['user_id']
    user_email = auth.current_user()['email']
    refresh_token = auth.current_user()['refresh_token']
    email_alert = auth.current_user()['email_alert']

    if refresh_token is None or refresh_token == '':
        return '', 403

    from autoupdate import autoupdate
    try:
        status, result = autoupdate(user_id, username, refresh_token, user_email, email_alert)
        if status == 'success':
            return {
                'alert_content': result
            }
        elif status == 'error':
            print (result)
            return result, 204
        else:
            return "unknown error", 204
    except:
        traceback.print_exc()
        return "exception in autoupdate", 204
# app.run()
