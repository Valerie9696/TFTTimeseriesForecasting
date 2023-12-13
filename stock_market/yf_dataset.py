import pandas as pd
import yfinance as yf

TODAY = pd.to_datetime("today").date()
START = (TODAY - pd.DateOffset(days=29)).date()

# Reference: https://stackoverflow.com/a/48131963/16051077
d1 = pd.date_range(start=START, end=TODAY, freq="7D")
d2 = d1.shift(6, freq="d")
# fix end date (make sure latest end_date it doesn't go over end_date)
d2 = list(d2)[:-1] + [min(d2[-1], pd.Timestamp(TODAY))]

dates = pd.DataFrame(
    data=dict(start_date=d1, end_date=d2), columns=("start_date", "end_date")
)

df_list = []
for i in dates.index:
    start = dates.at[i, "start_date"]
    end = dates.at[i, "end_date"]

    tickers = ["TSLA", "MSFT", "AMZN"]

    df = yf.download(tickers, start=start, end=end, interval="1m")["Adj Close"]
    df_list.append(df)

history = pd.concat(df_list)