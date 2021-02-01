import requests
import pandas
import datetime
import redis
import os
import json
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from database import load_models
from utils import convert_frequency
from tsapi import load_ts_prices
from tsapi import get_access_token
from engine import train_and_test
from database import push_alert
from constants import sendgrid_api_key


test_mode = False

files = [['BCRaw', 'BTC']]
mail_from = 'david@buildalpha.com'

def send_email(mail_to, email_html, email_text):
  message = Mail(
    from_email=mail_from,
    to_emails=mail_to,
    subject='New prediction update',
    html_content=email_html)

  try:
      sg = SendGridAPIClient(sendgrid_api_key)
      response = sg.send(message)
      print(response.body)
  except Exception as e:
      print('Exception: ' + str(e))

def autoupdate(user_id, username, refresh_token, user_email, email_alert):
  result = load_models(user_id)
  if not result['success']:
    return "error", "load_models failed"
  
  predictions = []
  for model in result['models']:
    symbol = model["symbol"]
    frequency = model["frequency"]
    start_date = model["start_date"] if "start_date" in model else "10-01-2010"
    if symbol is None or frequency is None:
      continue
    unit, interval = convert_frequency(frequency)
    filename = "TSData_" + symbol + "_" + model["frequency"] + "_" + start_date
    
    access_token = get_access_token(refresh_token)
    if access_token is None:
      return "error", "get_access_token failed"
    rd = redis.from_url(os.environ.get("REDIS_URL"))
    rd_file = rd.get(filename)
    if rd_file is not None and rd_file != 'failed' and rd_file != 'waiting':
      pd = pandas.read_msgpack(rd_file)

      last_date = pd['Date'][pd.index[-1]]
      last_time = pd['Time'][pd.index[-1]]
      if last_date is None or last_time is None:
        pd = load_ts_prices(access_token, symbol, frequency, start_date)
      else:
        td = datetime.timedelta(days=1) if unit == "Daily" or unit == "Weekly" or unit == "Monthly" else datetime.timedelta(minutes=interval)
        cur_date = datetime.datetime.strptime(last_date + ' ' + last_time, '%m/%d/%Y %H:%M:%S') + td
        from_date = cur_date.strftime('%m-%d-%Yt%H:%M:%S')
        to_date = datetime.datetime.now().strftime('%m-%d-%Yt%H:%M:%S')
        print("now: " + to_date)
        print("the last time: " + from_date)
        if datetime.datetime.now() > cur_date:
          r = requests.get('https://api.tradestation.com/v2/stream/barchart/' + symbol + '/' + str(interval) + '/' + unit + '/' + from_date + '/' + to_date + '?access_token=' + access_token)
          if r.status_code == 200:
            from utils import parse_prices
            new_pd = parse_prices(r.text)
            pd = pandas.concat([pd, new_pd], ignore_index=True)
          rd.set(filename, pd.to_msgpack(compress='zlib'))

          pd = pandas.read_msgpack(rd.get(filename))
        else:
          continue
    else:
      pd = load_ts_prices(access_token, symbol, frequency, start_date)
    
    if pd is None:
      return "error", "can't get data from tradestation"
    
    options = json.loads(model['model_options'])      
    try:
      res_data_set = train_and_test(pd, options['transforms'], options['parameters'], optimize=False)
      predict_data = res_data_set[0][0][2]
      predict_data_val = predict_data[len(predict_data)-1]
      add_alert = False
      alert_condition = options['parameters']['alertCondition'] if 'alertCondition' in options['parameters'] else 'equal'
      alert_threshold = float(options['parameters']['alertThreshold']) if 'alertThreshold' in options['parameters'] else 0
      if alert_condition == "equal":
        add_alert = True if predict_data_val == alert_threshold else False
      elif alert_condition == "above":
        add_alert = True if predict_data_val > alert_threshold else False
      elif alert_condition == "below":
        add_alert = True if predict_data_val < alert_threshold else False
      if add_alert:
        predictions.append([
          datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'),
          model['model_name'],
          options['parameters']['trainLabel'],
          predict_data_val
        ])
    except:
      traceback.print_exc()
      continue

  if len(predictions) > 0:
    result_email_html = '<html><body><h3>New prediction based on your models</h3><table><tr><td>Date</td><td>Model Name</td><td>Prediction Label</td><td>Prediction</td></tr>'
    result_email_text = 'New predictions based on your models!\n'
    for date, name, label, value in predictions:
      value_str = ('%.4f' % value)
      row_html = '<tr><td>' + date + '</td><td>' + name + '</td><td>' + label + '</td><td>' + value_str + '</td></tr>'
      result_email_html = result_email_html + row_html
      result_email_text = result_email_text + date + ': ' + name + '\'s ' + label + ' = ' + value_str + '\n' 
    result_email_html = result_email_html + '</table></body></html>'
    print(result_email_html)
  
    if email_alert:
      send_email(user_email, result_email_html, result_email_text)
    
    return "success", result_email_html
  else:
    return "error", "no predictions to return"

if __name__ == "__main__":
  from database import load_users
  result = load_users()
  if result['success']:
    for user in result['users']:
      if user['model_count'] > 0:
        try:
          autoupdate(user['user_id'], user['username'], user['refresh_token'], user['email'], user['email_alert'])
        except Exception as e:
          print(e)