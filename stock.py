import numpy as np
import tensorflow as tf
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data


class Stock:
    # intervals:
    #     minute=60*60
    #     sec=60
    #     hour=60*60*60
    #     day=60*60*60*24
    #     week=60*60*60*24*7
    #     month=60*60*60*24*30
    
    sec = str(60)
    minute = str(sec*60)
    hour = str(minute*60)
    day = str(24*hour)
    week = str(day*7)
    month = str(day*30)

    intervals = [sec, minute, hour, day, month]
    period = ["1Y", "1M", "1D"]

    i_choice = 0    #default 1 sec
    p_choice = 0    #default 1 Year    
    param_choice = None   #param choice default none

    #time step
    time_step = None

    # Dow Jones
    param_dji = {
        'q': ".DJI", # Stock symbol (ex: "AAPL")
        'i': intervals[i_choice], 
        'x': "INDEXDJX", # Stock exchange symbol on which stock is traded (ex: "NASD")
        'p': period[p_choice] # Period (Ex: "1Y" = 1 year)
    }

    # Apple
    param_aapl = {
        'q': "AAPL", # Stock symbol (ex: "AAPL")
        'i': intervals[i_choice], # Interval size in seconds ("86400" = 1 day intervals)
        'x': "NASDAQ", # Stock exchange symbol on which stock is traded (ex: "NASD")
        'p': period[p_choice] # Period (Ex: "1Y" = 1 year)
    }

    # Facebook
    param_fb = {
        'q': "FB", # Stock symbol (ex: "AAPL")
        'i': intervals[i_choice], # Interval size in seconds ("86400" = 1 day intervals)
        'x': "NASDAQ", # Stock exchange symbol on which stock is traded (ex: "NASD")
        'p': period[p_choice] # Period (Ex: "1Y" = 1 year)
    }

    params = [param_dji, param_aapl, param_fb]

    def __init__(self,
            param_choice, interval_choice,
            period_choice,
            configs,
            last_cell_states=10):
        self.i_choice = interval_choice
        self.p_choice = period_choice
        self.param_choice = self.params[param_choice]
        self.time_step = configs.time_steps
        self.input_size = configs.input_size
        self.last_cell_states = last_cell_states

    def get_data(self):
        df = get_price_data(self.param_choice)

        seq = np.matrix(df['Close'].values).transpose()
        seq = [float(np.array(i)) for i in seq]

        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size]) for i in range(len(seq) // self.input_size)]

        # Split into groups of `num_steps`
        X = np.array([seq[i: i + self.time_step] for i in range(len(seq) - self.time_step)])
        y = np.array([seq[i + self.time_step] for i in range(len(seq) - self.time_step)])
        return X, y

    def generate_one_epoch(self, batch_size):
        X, y = self.get_data()
        num_batches = int(len(X)) // batch_size
        if batch_size * num_batches < len(X):
            num_batches += 1

        batch_indices = range(num_batches)
        for j in batch_indices:
            batch_X = X[j * batch_size: (j+1) * batch_size]
            batch_y = y[j * batch_size: (j+1) * batch_size]
            yield  np.array( batch_X ), np.array( batch_y )
