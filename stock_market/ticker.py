import yfinance as yf


class Ticker:
    def __init__(self, ticker_name):
        try:
            self.name = ticker_name
            self.ticker = yf.Ticker(ticker_name)
            self.info = self.ticker.info
            self.chart = self.get_chart()
        except:
            self.ticker = None

    def get_stock_info(self):
        stock_info = None
        stock_info_json = None
        stock_descr = None
        stock_descr_json = None
        try:
            stock_descr = ["city", "country", "industry", "sector", "price-earnings-ratio", "institutional holders"]    #add recommendations when yfinance has recovered from json decryption issues
            stock_descr_json = ["dividends", "institutional_holders"]
            stock_info = (self.info['city'], self.info['country'], self.info['industry'], self.info['sector'],
                          self.info['forwardPE'])
            stock_info_json = (self.ticker.dividends.to_json(), self.ticker.get_institutional_holders().to_json())
        except:
            stock_info = None
        return "found", stock_descr, stock_descr_json, stock_info, stock_info_json

    def get_chart(self):
        chart_data = self.ticker.history(period="max")
        print(chart_data)
        return 0

t =Ticker("MSFT")
a=0

ticker_symbol = "AAPL"

# Get the data for the stock
stock_data = yf.Ticker(ticker_symbol)

# Get the historical data for the stock, including volume
historical_data = stock_data.history(start="2022-04-20", end="2022-04-26", interval="1d")

# Get the daily traded volume for the specified period of time
daily_volumes = historical_data["Volume"]

print("Daily traded volumes for " + ticker_symbol + " between 2022-04-20 and 2022-04-26:")
print(daily_volumes)