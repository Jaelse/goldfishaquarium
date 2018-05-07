import numpy as np
import tensorflow as tf
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
import requests

class Stock:

    def __init__(self,
            param_choice, interval_choice, period_choice,
            configs,
            last_cell_states=10):
        self.time_step = configs.time_steps
        self.input_size = configs.input_size
        self.last_cell_states = last_cell_states

        self.configs = configs
        sec = 1
        minute = sec*60
        hour = minute*60
        day = 24*hour
        week = day*7
        month = day*30

        intervals = [sec, minute, hour, day, month]
        intervals = [str(i) for i in intervals]

        period = ["100Y", "1M", "1D"]
        # Dow Jones
        param_dji = {
            'q': ".DJI", # Stock symbol (ex: "AAPL")
            'i': intervals[interval_choice], 
            'x': "INDEXDJX", # Stock exchange symbol on which stock is traded (ex: "NASD")
            'p': period[period_choice] # Period (Ex: "1Y" = 1 year)
        }

        # Apple
        param_aapl = {
            'q': "AAPL", # Stock symbol (ex: "AAPL")
            'i': intervals[interval_choice], # Interval size in seconds ("86400" = 1 day intervals)
            'x': "NASDAQ", # Stock exchange symbol on which stock is traded (ex: "NASD")
            'p': period[period_choice] # Period (Ex: "1Y" = 1 year)
        }

        # Facebook
        param_fb = {
            'q': "FB", # Stock symbol (ex: "AAPL")
            'i': intervals[interval_choice], # Interval size in seconds ("86400" = 1 day intervals)
            'x': "NASDAQ", # Stock exchange symbol on which stock is traded (ex: "NASD")
            'p': period[period_choice] # Period (Ex: "1Y" = 1 year)
        }

        params = [param_dji, param_aapl, param_fb]
        self.param_choice = params[param_choice]
        
        self.get_data()

    def get_current_data(self):
        print ("Getting current data...")
        df = get_price_data(self.param_choice)

        ### Get data from alphavantage #######################################
        symbol = "MSFT"
        # symbol = "GOOG"
        # symbol = ".DJI"

        api_key = "W8Q9GP1SHM5OK409"

        function = "TIME_SERIES_INTRADAY"
        interval = "1min"
        url_interday = "https://www.alphavantage.co/query?function=%s&symbol=%s&outputsize=compact&interval=%s&apikey=%s" %(function, symbol, interval, api_key)

        r = requests.get(url_interday)

        data = r.json()
        close_list = []
        # print(data)
        for key in data:
            for inner_key in data[key]:
                record =  data[key][inner_key]
                if isinstance(record, dict):
                    close = record["4. close"]
                    close_list.append(close)
        close_list = close_list[:self.configs.time_steps]
        close_list.reverse()

        seq = np.matrix(close_list).transpose()
        seq = [float(np.array(i)) for i in seq]

        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size]) for i in range(len(seq) // self.input_size)]

        # Split into groups of `num_steps`

        X = seq

        # normalize = False
        normalize = True
        normFactorX = np.mean(X)
        self.normFactorX = normFactorX
        if normalize:
            X = X / normFactorX
        else:
            normFactorX = 1
        return [X]

    def get_data(self):
        print ("Getting data...")
        df = get_price_data(self.param_choice)

        ### Get data from alphavantage #######################################
        symbol = "MSFT"
        # symbol = "GOOG"
        # symbol = ".DJI"

        api_key = "W8Q9GP1SHM5OK409"

        function = "TIME_SERIES_INTRADAY"
        interval = "1min"
        url_interday = "https://www.alphavantage.co/query?function=%s&symbol=%s&interval=%s&outputsize=full&apikey=%s" %(function, symbol, interval, api_key)

        function = "TIME_SERIES_DAILY"
        url_daily = "https://www.alphavantage.co/query?function=%s&symbol=%s&outputsize=full&apikey=%s" %(function, symbol, api_key)

        r = requests.get(url_interday)
        # r = requests.get(url_daily)

        data = r.json()
        close_list = []
        # print(data)
        for key in data:
            for inner_key in data[key]:
                record =  data[key][inner_key]
                if isinstance(record, dict):
                    close = record["4. close"]
                    # print(inner_key, close)
                    close_list.append(close)
        close_list.reverse() # get data in ascending order.
        self.other_input = close_list[:50]
        # print(close_list)
        # print(self.other_input)
        #######################################################################

        # seq = np.matrix(df['Close'].values).transpose()
        seq = np.matrix(close_list).transpose()
        seq = [float(np.array(i)) for i in seq]

        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size]) for i in range(len(seq) // self.input_size)]

        # Split into groups of `num_steps`

        X = np.array([seq[i: i + self.time_step] for i in range(len(seq) - self.time_step)])
        y = np.array([seq[i + self.time_step] for i in range(len(seq) - self.time_step)])

        # normalize = False
        normalize = True
        normFactorX = np.mean(X)
        normFactory = np.mean(y)
        self.normFactorX = normFactorX
        self.normFactory = normFactory
        if normalize:
            X = X / normFactorX
            y = y / normFactory
        else:
            normFactorX = 1
            normFactory = 1

        train_size = len(X) - len(X)//20;
        # train_size = len(X) - self.time_step * 2;

        self.train_X, self.test_X = X[:train_size], X[train_size:]
        self.train_y, self.test_y = y[:train_size], y[train_size:]
        
        return self.train_X, self.train_y, self.test_X, self.test_y

    # Generator which gives yields batches for a single epoch.
    def generate_one_epoch(self, batch_size):
        # train_X, train_y, test_x, test_y = self.get_data()
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = range(num_batches)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j+1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j+1) * batch_size]
            yield  np.array( batch_X ), np.array( batch_y )
