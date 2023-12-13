import yfinance as yf
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as chart
import matplotlib.pyplot as plt
import pandas_ta as pta

def gather(folder_path):
    full_df = pd.DataFrame()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if (filename != 'temp.csv' and filename != 'ohlc.csv') and os.path.isfile(file_path):
            df = pd.read_csv(filepath_or_buffer=file_path)
            symbol = filename.split('.')[0]
            print(type(symbol), symbol)
            df['Symbol'] = '.'+symbol
            df['StatSymbol'] = '.'+symbol
            hour_of_day = []
            hours_from_start = []
            day_of_week = []
            day_of_month = []
            week_of_year = []
            hos = 0
            s = df['Symbol'].iloc[0]
            for symbol in df['Symbol']:
                if symbol == s:
                    hos += 1
                    hours_from_start.append(hos)
                else:
                    s = symbol
                    hos = 1
            for date in df['Datetime']:
                parts = date.split(' ')
                date = dt.strptime(parts[0], '%Y-%m-%d')
                day_of_week.append(date.weekday())
                day_of_month.append(date.day)
                week_of_year.append(date.isocalendar().week)
                time = parts[1]
                hour = int(time[0:2])
                hour_of_day.append(hour)
            df['HourOfDay'] = hour_of_day
            df['HoursFromStart'] = hours_from_start
            df['DayOfWeek'] = day_of_week
            df['DayOfMonth'] = day_of_month
            df['WeekOfYear'] = week_of_year
            df['OpenCloseDifference'] = df['Open'] - df['Close']
            df['HighLowDifference'] = df['High'] - df['Low']
            df['RSI'] = pta.rsi(df['Close'], length=14)
            df['RSI'].iloc[0:14] = 40
            a= pta.macd(close=df['Close'])
            b = pta.bbands(close=df['Close'])
            df['ATR'] = pta.atr(high=df['High'], low=df['Low'], close=df['Close'])
            df['ATR'].iloc[0:14] = 0.5
            df.reset_index(drop=True)
            print(help(pta.bbands))
            df = df.iloc[:, 1:]
            print(symbol)
            if not os.path.exists(os.path.join(folder_path, 'singles')):
                os.mkdir(os.path.join(folder_path, 'singles'))
            df.to_csv(os.path.join(folder_path, 'singles', symbol+'.csv'))
            full_df = pd.concat([full_df, df])
    full_df.to_csv(os.path.join(folder_path, 'ohlc.csv'))

gather(os.path.join('ticker_data', 'hourly'))