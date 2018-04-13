import numpy as np
import tensorflow as tf
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data


class Stock:
        
    #intervals 
    #sec=60, minute=60*60, hour=60*60*60, day=60*60*60*24, week=60*60*60*24*7, month=60*60*60*24*30
    
    sec = str(60)
    minute = str(sec*60)
    hour = str(minute*60)
    day = str(24*hour)
    week = str(day*7)
    month = str(day*30)

    intervals = [sec,minute,hour,day,month]
    period = ["1Y","1M","1D"]

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

    params = [param_dji,param_aapl,param_fb]

    def __init__(self,param_choice,i_choice,p_choice,time_step=30):
        self.i_choice = i_choice
        self.p_choice = p_choice
        self.param_choice = self.params[param_choice]
        self.time_step = time_step

    def print(self):
        print(self.i_choice)
        print(self.p_choice)
        print(self.param_choice)

    def get_data(self):
        df = get_price_data(self.param_choice)

        stocks = np.matrix(df.values)

        close = np.matrix(df['Close'].values).transpose()
        
        #number of data elements
        len = close.shape[0]

        #sequence size
        sequence = self.time_step

        #finding how many can we make
        row = int(len/sequence)

        #number of rows to remove
        remainder = len - (sequence * row)

        #now deleting the first 'remainder' number of rows
        close = np.delete(close, np.s_[:remainder]).transpose()

        a = np.reshape(close,(1,np.shape(close)[0]))

        close = np.reshape(close,(row,self.time_step))
        
        close = np.split(close,2)

        return close[0],close[1]