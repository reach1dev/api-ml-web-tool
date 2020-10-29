import pandas
import datetime
import json

def convert_frequency(frequency: str):
  unit = "Minute"
  interval = 1
  if frequency.endswith("m"):
    interval = int(frequency.replace("m", ""))
  elif frequency.endswith("h"):
    interval = int(frequency.replace("h", "")) * 60
  
  if "1D" == frequency:
    unit = "Daily"
  elif "1W" == frequency:
    unit = "Weekly"
  elif "1M" == frequency:
    unit = "Monthly"
  elif frequency.endswith("m") or frequency.endswith("h"):
    unit = "Minute"
  else:
    return None
  return unit, interval


def parse_prices(lines: str):
  data = []
  try:
    for line in lines.splitlines():
      d = parse_prices_line(line)
      if d is None:
        break
      data.append(d)
  except:
    pass
  df = pandas.DataFrame(data, columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol', 'OI'])
  return df


def parse_prices_line(line: str):
  if 'END' in line:
    return True
  if line is None or line == '':
    return False
  try:
    prices = json.loads(line)
  except:
    return False
  ts = float(prices['TimeStamp'].replace('/Date(', '').replace(')/', '')) / 1000
  if ts < 0:
    return False
  dt = datetime.datetime.fromtimestamp(ts)
  return [
    dt.strftime('%m/%d/%Y'),
    dt.strftime('%H:%M:%S'),
    float(prices['Open']),
    float(prices['High']),
    float(prices['Low']),
    float(prices['Close']),
    float(prices['TotalVolume']),
    float(prices['OpenInterest']),
  ]
