import yfinance as yf
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as chart
import matplotlib.pyplot as plt

ticker_list = ['ALV.DE', 'BAS.DE', 'BMW.DE', 'HEN.DE', 'SIE.DE', 'ADS.DE', 'CBK.DE', 'DTE.DE', 'EOAN.DE', 'HEI.DE', 'IFX.DE'
               , 'MRK.DE', 'SAP.DE', 'VNA.DE', 'RWE.DE', 'RHM.DE', 'HNR1.DE', '1COV.DE', 'SY1.DE', 'AIR.DE']
#Allianz, BASF, BMW, Henkel, Siemens, Aidas, Commerzbank, Deutsche Telekom, E.ON, HeidelbergCement, Infineon, Merck
#SAP, Vonovia, RWE, Rheinmetall, Hannover Rück, Covestro, Symrise, Airbus


# given a list of ticker symbols, make a dataset that corresponds to the given time interval and frequency
# this function updates existing files and appends new data rows if it is run again
def make_dataset(ticker_list, interval, freq):
    #valid for freq is minutely, hourly
    # Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    if freq == 'minutely':
        d = timedelta(days=6)
    elif freq == 'hourly':
        d = timedelta(days=729)
    cur_date = dt.now()
    for tcr in ticker_list:
        try:
            ticker = yf.Ticker(ticker=tcr)
            data = ticker.history(interval=interval, start=cur_date-d, end=cur_date.now())
            save('temp.csv', freq, data)
            prev = load(tcr + '.csv', freq)
            data = load('temp.csv', freq)
            if not prev.empty:
                data = pd.concat([prev, data]).drop_duplicates(subset='Datetime', inplace=False).reset_index(drop=True)
                data = data.iloc[: , 1:]
            save(tcr + '.csv', freq, data)
        except:
            print('An error occurred.')


def load(filename, interval):
    data = pd.DataFrame()
    if os.path.exists(os.path.join('ticker_data', interval, filename)):
        data = pd.read_csv(os.path.join('ticker_data', interval, filename))
    return data


# saves given dataframe 'data' at filename in folder that is named like the interval (hourly, minutely, daily)
def save(filename, interval, data):
    if not os.path.isdir('ticker_data'):
        os.mkdir('ticker_data')
    if not os.path.isdir(os.path.join('ticker_data',interval)):
        os.mkdir(os.path.join('ticker_data',interval))
    data.to_csv(os.path.join('ticker_data',interval, filename))


# makes candlestick chart of a dataframe and saves at given path with 'name'
def make_ohlc_chart(df, name, at_path):
    if not os.path.isdir(at_path):
        os.mkdir(at_path)
    df = df.reset_index()
    if 'Datetime' in df:
        fig = chart.Figure(data=[chart.Candlestick(x=df['Datetime'],
                                                   open=df['Open'],
                                                   high=df['High'],
                                                   low=df['Low'],
                                                   close=df['Close'])])
    else:
        fig = chart.Figure(data=[chart.Candlestick(x=df['Date'],
                                                   open=df['Open'],
                                                   high=df['High'],
                                                   low=df['Low'],
                                                   close=df['Close'])])
    fig.write_html(os.path.join(at_path, name+'.html'))


# get the maximum data that is available at a daily frequency and make candlestick charts
def get_max_daily_data():
    if not os.path.exists(os.path.join('ticker_data', 'daily')):
        os.mkdir(os.path.join('ticker_data', 'daily'))
    for tcr in ticker_list:
        d = yf.download(tcr, period='MAX', interval='1d')
        d.rename(columns={'Date':'Datetime'})
        make_ohlc_chart(d, tcr, os.path.join('ticker_data', 'daily', 'ohlc_plots'))
        save(tcr+'.csv', 'daily', d)


# plots all tickers in seperate open, high, low and close plots
def plot_all(directory, interval):
    #valid for interval is 60S, H
    keys = ['Open', 'High', 'Low', 'Close']
    df = pd.DataFrame()
    if interval == '60S':
        folder = 'minutely'
        idx = pd.date_range(start=dt.today()-timedelta(days=10), end=dt.today(), freq='min', tz='utc')
    elif interval == 'H':
        folder = 'hourly'
        idx = pd.date_range(start=dt.today() - timedelta(days=100), end=dt.today(), freq='min', tz='utc')
    if not os.path.isdir(os.path.join(directory, 'plots')):
        os.mkdir(os.path.join(directory, 'plots'))
    for key in keys:
        for csv in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, csv)):
                df = pd.read_csv(os.path.join('ticker_data', folder, csv))
                dates = df['Datetime']
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.set_index('Datetime')
                df.asfreq(freq=interval)
                label = csv.split('.')
                close = df[key].plot(marker=',', linestyle='-', grid=True, label=label[0])
        lgd = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.gcf().canvas.draw()
        invFigure = plt.gcf().transFigure.inverted()
        lgd_pos = lgd.get_window_extent()
        lgd_coord = invFigure.transform(lgd_pos)
        lgd_xmax = lgd_coord[1, 0]
        ax_pos = plt.gca().get_window_extent()
        ax_coord = invFigure.transform(ax_pos)
        ax_xmax = ax_coord[1, 0]
        shift = 1 - (lgd_xmax - ax_xmax)
        plt.gcf().tight_layout(rect=(0, 0, shift, 1))
        plt.xlabel('Datetime')
        plt.ylabel('Price')
        plt.title(key)
        plt.savefig(os.path.join('ticker_data', folder, 'plots', interval + key + '.png'))
        #plt.show()


# make plots and csv files with minutely, hourly, and daily data
make_dataset(ticker_list, '1m', 'minutely')
make_dataset(ticker_list, '1h', 'hourly')
get_max_daily_data()
plot_all(os.path.join('ticker_data','minutely'), '60S')
plot_all(os.path.join('ticker_data','hourly'), 'H')
intervals = ['minutely', 'hourly']
for itv in intervals:
    if not os.path.exists(os.path.join('ticker_data', itv, 'ohlc_plots')):
        os.mkdir(os.path.join('ticker_data', itv, 'ohlc_plots'))
    for file in os.listdir(os.path.join('ticker_data', itv)):
        if os.path.isfile(os.path.join('ticker_data', itv, file)):
            data = pd.read_csv(os.path.join('ticker_data', itv, file))
            name = file.split('.')[0]
            make_ohlc_chart(data, name, os.path.join('ticker_data', itv, 'ohlc_plots'))

