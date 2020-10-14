import requests
import pandas
import datetime
import redis
import os

def get_test_file():
  API_KEY = 'b28561460bd0ffb71e85eb2cae999490b650646068337239458d46917fde2c86'
  r = requests.get('https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=1&api_key=' + API_KEY)
  data = r.json()
  objects = data['Data']['Data']
  today_obj = objects[len(objects)-1]

  dt = datetime.datetime.fromtimestamp(today_obj['time'])
  file_path = 'tmp/BCRaw.txt'
  pd = pandas.read_csv(file_path)

  pd = pd.append(pandas.DataFrame([[
    dt.strftime('%m/%d/%Y'),
    dt.time().isoformat(),
    today_obj['open'],
    today_obj['high'],
    today_obj['low'],
    today_obj['close'],
    today_obj['volumefrom'],
    0
  ]], columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol', 'OI']))

  pack = pd.to_msgpack(compress='zlib')
  pd = pandas.read_msgpack(pack)

  return pd
