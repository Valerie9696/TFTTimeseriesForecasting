import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocessing
import tft_dataset


class Trade:
    """
    Class that organizes some very simple mock trading. Consists of the amount of stocks bought,
    their price at the time of the trade and whether they were bought or sold.
    """
    def __init__(self, amount, price, kind):
        self.amount = amount
        self.price = price
        self.kind = kind

    def get_total(self):
        """
        Helper to evaluate the final result of the trading script
        :return: total sum
        """
        if self.kind == 'bought':
            total = -self.price*self.amount
        else:
            total = self.price*self.amount
        return total


class Trader:
    """
    Class that organizes some very simple mock trading. Consists of an initial budget, a threshold which
    stops buying stocks indefinitely, keys which equal the companies ticker name, a portfolio and
    the final result.
    """
    def __init__(self, budget=100000, threshold=2, keys=[1,2,3,4]):
        self.budget = budget
        self.threshold = threshold
        self.keys = keys
        self.portfolio = self.setup_portfolio()
        self.final_result = 0

    def make_asset(self):
        """
        Each asset in the portfolio consist of the absolute amount of stocks bought, the current budget, a
        list of numbers for selling/ buying, a list of prices for selling/ buying, selling prices, and
        buying prices.
        :return:
        """
        asset = dict.fromkeys(['absolute_amount', 'budget', 'numbers', 'prices', 'buy_prices', 'sell_prices'])
        asset['absolute_amount'] = 0
        asset['budget'] = self.budget
        for i, _ in asset.items():
            if i == 'numbers' or i == 'prices' or i == 'buy_prices' or i == 'sell_prices':
                asset[i] = []
        return asset

    def setup_portfolio(self):
        """Makes a portfolio for each ticker in the given keys."""
        portfolio = dict.fromkeys(self.keys)
        for k, _ in portfolio.items():
            portfolio[k] = self.make_asset()
        return portfolio

    def sellout(self):
        """
        After ending the trading, sell everything that was bought in order to get the final budget.
        :return: the final budget
        """
        results = dict.fromkeys(self.keys)
        for key in self.portfolio:
            asset = self.portfolio[key]
            abs_amount = asset['absolute_amount']
            if abs_amount == 0:
                results[key] = asset['budget']
            elif abs_amount < 0:
                # buy until 0
                results[key] = asset['budget'] - abs_amount * asset['prices'][-1]
            elif abs_amount > 0:
                # sell until 0
                results[key] = asset['budget'] + abs_amount * asset['prices'][-1]
        self.final_result = results
        return results


if __name__ == '__main__':
    period = 'hourly'
    if not os.path.exists('TraderPlots'):
        os.mkdir('TraderPlots')
    if period == 'daily':
        df = preprocessing.DailyTFTDataset(path=os.path.join('ohlc', 'daily_ohlc.csv'))
        test_set = df.test_set.dropna()
        test = tft_dataset.make_daily_timeseries_dataset(dataset=test_set)
    elif period == 'hourly':
        df = preprocessing.TFTDataset(path=os.path.join('ohlc', 'ohlc.csv'))
        test_set = df.test_set.dropna()
        test = tft_dataset.make_timeseries_dataset(dataset=test_set)
    stock_names = test_set['StatSymbol'].unique()
    trader = Trader(keys=stock_names)
    buy_sell_number = 10
    if period == 'daily':
        predictions = pd.read_csv('daily_predictions.csv')
    elif period == 'hourly':
        predictions = pd.read_csv('predictions.csv')
    predictions = predictions.iloc[: , 1:]
    plotter = {}
    for i in range(10, len(test_set)-(len(test_set)-len(predictions))):
        try:
            test_row = test_set.iloc[i]
            past = test_set.iloc[i-10:i]
            past_mean = np.mean(past['Close'])
            symbol = test_row['StatSymbol']
            price = test_row['Close']
            mean = predictions.iloc[i].mean()
            cur_asset = trader.portfolio[symbol]
            #mean = np.mean(pred_row)
            #sell
            if len(cur_asset['buy_prices']) > 0:
                check_bigger_last = mean > cur_asset['buy_prices'][-1]
                buy_diff = price > cur_asset['buy_prices'][-1]-3
            else:
                check_bigger_last = True
                buy_diff = True
            if len(cur_asset['sell_prices']) > 0:
                check_smaller_last = mean < cur_asset['sell_prices'][-1]
                sell_diff = price > cur_asset['buy_prices'][-1] + 3
            else:
                check_smaller_last = True
                sell_diff = True
            #sell
            if mean < price and check_bigger_last and sell_diff and past_mean > price:
                cur_asset = trader.portfolio[symbol]
                if cur_asset['absolute_amount'] > 9:
                    cur_asset['budget'] = cur_asset['budget']+buy_sell_number*price
                    cur_asset['absolute_amount'] = cur_asset['absolute_amount']-buy_sell_number
                    cur_asset['numbers'].append(buy_sell_number)
                    cur_asset['prices'].append(price)
                    cur_asset['sell_prices'].append(price)
            #buy
            elif past_mean < price and price < mean and check_smaller_last and buy_diff:
                cur_asset = trader.portfolio[symbol]
                cur_asset['budget'] = cur_asset['budget'] - buy_sell_number * price
                cur_asset['absolute_amount'] = cur_asset['absolute_amount'] + buy_sell_number
                cur_asset['numbers'].append(buy_sell_number)
                cur_asset['prices'].append(price)
                cur_asset['buy_prices'].append(price)
            if symbol not in list(plotter.keys()):
                plotter[symbol] = {'budget': [cur_asset['budget']], 'absolute_amount': [cur_asset['absolute_amount']]}
            else:
                plotter[symbol]['budget'].append(cur_asset['budget'])
                plotter[symbol]['absolute_amount'].append(cur_asset['absolute_amount'])
        except:
            print(i)
    for key in plotter:
        cur_asset = plotter[key]
        plt.plot(cur_asset['budget'])
        plt.title(key+' budget')
        plt.show()
        plt.savefig(os.path.join('TraderPlots', key + ' budget.png'))
        plt.plot(cur_asset['absolute_amount'])
        plt.title(key + ' amount')
        plt.show()
        plt.savefig(os.path.join('TraderPlots', key + ' amount.png'))
    trader.sellout()
    average = sum(trader.final_result.values())/len(trader.final_result.values())
    print('Start budget: ')
    print(trader.budget)
    print('Average final result:')
    print(average)