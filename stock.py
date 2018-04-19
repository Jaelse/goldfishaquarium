import numpy as np
import tensorflow as tf
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data

class Stock:
    def __init__(self,
            param_choice, interval_choice, period_choice,
            configs):
        self.time_step = configs.time_steps
        self.input_size = configs.input_size

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

    def get_data(self):
        df = get_price_data(self.param_choice)

        close = np.matrix(df['Close'].values).transpose()

        for_input = np.reshape(close[0:30],(1,30))

        i = 0
        inputs = []
        targets = []

        len = np.shape(close)[0]
        
        can_make_set = True

        while(can_make_set):
            if  i+self.time_step+1 < len:
                for_input = np.reshape(close[i:i+self.time_step],(1,self.time_step))
                inputs.append(np.matrix.tolist(for_input))
                i = i+self.time_step
                for_targets = np.reshape(close[i:i+1],(1,1))
                targets.append(np.matrix.tolist(for_targets))
            else:    
                break
        
        inputs = np.reshape(inputs,(np.shape(inputs)[0],np.shape(inputs)[2]))
        targets = np.reshape(targets,(np.shape(targets)[0],np.shape(targets)[2]))
        
        self.targets = targets
        return inputs,targets
