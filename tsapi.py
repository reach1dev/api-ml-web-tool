import requests
import json
import datetime
import pandas
import http.client
import traceback
from utils import parse_prices
from utils import parse_prices_line
from constants import ts_client_secret
from constants import ts_client_id
from constants import ts_api_auth_url


def get_token(token, auth_code):
  res = requests.post(ts_api_auth_url, data = {
    "grant_type": "authorization_code",
    "client_id": ts_client_id,
    "redirect_uri": "https://api-ml-web-tool.herokuapp.com/account/tsapi_callback/" + token,
    "client_secret": ts_client_secret,
    "code": auth_code,
    "response_type": "token"
  })
  
  if res.status_code == 200:
    return res.json()
  else:
    print (res)
  return None


def get_access_token(refresh_token):
  res = requests.post(ts_api_auth_url, data = {
    "grant_type": "refresh_token",
    "client_id": ts_client_id,
    "redirect_uri": "https://ml-web-tool.herokuapp.com/",
    "client_secret": ts_client_secret,
    "refresh_token": refresh_token,
    "response_type": "token"
  })
  
  if res.status_code == 200:
    return res.json()['access_token']
  return None


def load_ts_prices(access_token, symbol, frequency, start_date):
  from utils import convert_frequency
  unit, interval = convert_frequency(frequency)
  
  symbol = symbol.replace('aATt', '@')
  dt_format = '%m-%d-%Y'
  today = datetime.datetime.today()
  today_str = today.strftime(dt_format)
  read_date = datetime.datetime.strptime(start_date, dt_format)
  data = []
  do_break = False
  while not do_break:
    read_date_str = read_date.strftime(dt_format)
    next_date_str = today_str
    if (today-read_date).days > 365:
      next_date = read_date + datetime.timedelta(days=365)
      next_date_str = next_date.strftime(dt_format)
      read_date = next_date + datetime.timedelta(days=1)
    else:
      do_break = True
      
    url = "https://api.tradestation.com/v2/stream/barchart/" + symbol + "/" + str(interval) + "/" + unit + "/" + read_date_str + "/" + next_date_str + "?access_token=" + access_token
    try:
      res = requests.get(url, stream=True)    
      if res.status_code == 200:      
        for line in res.iter_lines():
          if line is None or len(line) == 0:
            continue
          d = parse_prices_line(line.decode("utf-8"))
          if d is None:
            continue
          if len(d) == 0:
            break
          data.append(d)
      else:
        print("status_code = " + str(res.status_code))
        continue
    except:
      traceback.print_exc()
      continue
  if len(data) == 0:
    return None
  df = pandas.DataFrame(data, columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol', 'OI'])
  return df


if __name__ == "__main__":
  from constants import test_key
  access_token = get_access_token(test_key)
  df = load_ts_prices(access_token, '@BTC', '15m', '10-26-2020')
  print(df)