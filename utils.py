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
  for line in lines.splitlines():
    if 'END' in line or line is None or line == "":
      break
    try:
      prices = json.loads(line)
    except:
      break
    ts = float(prices['TimeStamp'].replace('/Date(', '').replace(')/', '')) / 1000
    if ts < 0:
      continue
    dt = datetime.datetime.fromtimestamp(ts)
    data.append([
      dt.strftime('%m/%d/%Y'),
      dt.strftime('%H:%M:%S'),
      float(prices['Open']),
      float(prices['High']),
      float(prices['Low']),
      float(prices['Close']),
      float(prices['TotalVolume']),
      float(prices['OpenInterest']),
    ])
  df = pandas.DataFrame(data, columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol', 'OI'])
  return df
