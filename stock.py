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

    def __init__(self,param_choice,i_choice,p_choice):
        self.i_choice = i_choice
        self.p_choice = p_choice
        self.param_choice = self.params[param_choice]

    def print(self):
        print(self.i_choice)
        print(self.p_choice)
        print(self.param_choice)

    def get_data(self):
        df = get_price_data(self.param_choice)

        stocks = np.matrix(df.values)

        x = np.matrix(df['Open'].values).transpose()
        y = np.matrix(df['Close'].values).transpose()

        #number of data elements
        len = x.shape[0]
        
        #sequence size
        sequence = 60

        #finding how many can we make
        row = int(len/sequence)

        #number of rows to remove
        remainder = len - (sequence * row)

        #now deleting the first 'remainder' number of rows
        x = np.delete(x, np.s_[:remainder]).transpose()
        y = np.delete(y, np.s_[:remainder]).transpose()

        return x,y


#TODO make the X,Y by just using close values

