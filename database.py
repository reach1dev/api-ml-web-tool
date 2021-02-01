import os
import psycopg2
import psycopg2.extras
import json
import hmac
from datetime import datetime
import jwt
from constants import DB_SECRET_KEY


DATABASE_URL = os.environ['DATABASE_URL']


def time_expired(date):
  d1 = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
  d2 = datetime.now()
  return abs((d2 - d1).seconds) > 12*3600


def save_model(user_id, model_name, symbol, frequency, start_date, transforms, parameters):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    model_options = json.dumps({
      'transforms': transforms,
      'parameters': parameters
    })
    cur.execute("INSERT INTO predict_models (user_id, model_name, symbol, frequency, start_date, model_options) VALUES (%s, %s, %s, %s, %s, %s)", (user_id, model_name, symbol, frequency, start_date, model_options))
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print(e)
    return False


def push_alert(username, alert_content):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    alert_time = datetime.now().isoformat()
    print('push_alert >> ' + alert_time)
    cur.execute("INSERT INTO alerts (user_name, alert_content, alert_time, is_pushed) VALUES (%s, %s, %s, %s)", (username, alert_content, alert_time, 't'))
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print(e)
    return False


def update_model(model_id,transforms, parameters, symbol, frequency, start_date):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    model_options = json.dumps({
      'transforms': transforms,
      'parameters': parameters
    })
    cur.execute("UPDATE predict_models SET model_options=%s, symbol=%s, frequency=%s, start_date=%s WHERE model_id=%s", (model_options, symbol, frequency, start_date, model_id))
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print(e)
    return False


def remove_model(model_id):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    cur.execute("DELETE FROM predict_models WHERE model_id=%s", [model_id])
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print(e)
    return False


def load_models(user_id):
  conn = psycopg2.connect(DATABASE_URL, sslmode='require')
  cur = conn.cursor()
  cur.execute("SELECT model_id, model_name, model_options, symbol, frequency, start_date FROM predict_models WHERE user_id=%s", [user_id])
  models = cur.fetchall()
  cur.close()
  conn.close()
  res = []
  for model in models:
    res.append({
      'model_id': model[0],
      'model_name': model[1],
      'model_options': model[2],
      'symbol': model[3],
      'frequency': model[4],
      'start_date': model[5]
    })
  return {
    'success': True,
    'models': res
  }


def load_users():
  conn = psycopg2.connect(DATABASE_URL, sslmode='require')
  cur = conn.cursor()
  cur.execute("SELECT a.user_id, a.email_address, a.user_name, a.tsapi_refresh_token, a.web_alert, a.email_alert, count(b.model_id) as model_count FROM users a LEFT JOIN predict_models b ON a.user_id=b.user_id GROUP BY a.user_id")
  users = cur.fetchall()
  cur.close()
  conn.close()
  res = []
  for user in users:
    res.append({
      'user_id': user[0],
      'email': user[1],
      'username': user[2],
      'refresh_token': user[3],
      'web_alert': user[4],
      'email_alert': user[5],
      'model_count': user[6]
    })
  return {
    'success': True,
    'users': res
  }


def find_user(username):
  conn = psycopg2.connect(DATABASE_URL, sslmode='require')
  cur = conn.cursor()
  cur.execute("SELECT user_id, user_name, email_address, full_name, tsapi_refresh_token, web_alert, email_alert FROM users WHERE user_name=%s", [username])
  users = cur.fetchall()
  cur.close()
  conn.close()
  if len(users) == 1:
    return {
      'user_id': users[0][0],
      'username': users[0][1],
      'email': users[0][2],
      'full_name': users[0][3],
      'refresh_token': users[0][4],
      'web_alert': users[0][5],
      'email_alert': users[0][6]
    }
  return None


def check_user(username, password):
  conn = psycopg2.connect(DATABASE_URL, sslmode='require')
  cur = conn.cursor()
  cur.execute("SELECT email_address, full_name, web_alert, email_alert FROM users WHERE user_name=%s AND password_hash=%s", [username, password])
  users = cur.fetchall()
  cur.close()
  conn.close()
  if len(users) == 1:
    return {
      'email': users[0][0],
      'fullName': users[0][1],
      'webAlerts': users[0][2],
      'emailAlerts': users[0][3]
    }
  return None


def clear_user_token(username):
  conn = psycopg2.connect(DATABASE_URL, sslmode='require')
  cur = conn.cursor()
  cur.execute("UPDATE users SET tsapi_refresh_token='', tsapi_access_token='' WHERE user_name=%s", [username])
  conn.commit()
  cur.close()
  conn.close()

def update_user(username, email, full_name):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    cur.execute("UPDATE users SET email_address=%s, full_name=%s WHERE user_name=%s", [email, full_name, username])
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print(str(e))
    return False


def update_user_alerts(username, web_alert, email_alert):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    cur.execute("UPDATE users SET web_alert=%s, email_alert=%s WHERE user_name=%s", [web_alert, email_alert, username])
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print(str(e))
    return False


def get_alerts_by_username(username):
  conn = psycopg2.connect(DATABASE_URL, sslmode='require')
  cur = conn.cursor()
  cur.execute("SELECT alert_id, alert_content, alert_time FROM alerts WHERE is_pushed='f' AND user_name=%s", [username])
  result = cur.fetchall()
  cur.close()
  conn.close()
  
  alerts = []
  for res in result:
    alerts.append({
      'alert_id': res[0][0],
      'alert_content': res[0][1],
      'alert_time': res[0][2],
    })
  return alerts


def pop_user_alerts(alert_id):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET is_pushed='t' WHERE alert_id=%s", [alert_id])
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print(str(e))
    return False


def update_user_tsapi_tokens(username, access_token, refresh_token):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    cur.execute("UPDATE users SET tsapi_access_token=%s, tsapi_refresh_token=%s WHERE user_name=%s", [access_token, refresh_token, username])
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print(str(e))
    return False


def get_user_tsapi_access_token(username):
  conn = psycopg2.connect(DATABASE_URL, sslmode='require')
  cur = conn.cursor()
  cur.execute("SELECT tsapi_refresh_token FROM users WHERE user_name=%s", [username])
  users = cur.fetchall()
  cur.close()
  conn.close()
  if len(users) == 1:
    refresh_token = users[0][0]
    from tsapi import get_access_token
    return get_access_token(refresh_token)
  return None

def upload_users(users):
  try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    psycopg2.extras.execute_values(cur, 'INSERT INTO users(user_name, password_hash) values %s ON CONFLICT (user_name) DO UPDATE SET password_hash=EXCLUDED.password_hash', users, template=None, page_size=100)
    conn.commit()
    cur.close()
    conn.close()
    return True
  except Exception as e:
    print (e)
    return False


def encode_auth_token(username):
  """
  Generates the Auth Token
  :return: string
  """
  try:
    payload = {
        'expire_time': datetime.datetime.utcnow() + datetime.timedelta(days=0, seconds=5),
        'created_time': datetime.datetime.utcnow(),
        'username': username
    }
    return jwt.encode(
        payload,
        DB_SECRET_KEY,
        algorithm='HS256'
    )
  except Exception as e:
    return e


def decode_auth_token(auth_token):
  """
  Decodes the auth token
  :param auth_token:
  :return: integer|string
  """
  try:
    payload = jwt.decode(auth_token, DB_SECRET_KEY)
    return payload['sub']
  except jwt.ExpiredSignatureError:
    return 'Signature expired. Please log in again.'
  except jwt.InvalidTokenError:
    return 'Invalid token. Please log in again.'