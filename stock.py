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

    def __init__(self,param_choice,i_choice,p_choice,time_step=30,last_cell_states=10):
        self.i_choice = i_choice
        self.p_choice = p_choice
        self.param_choice = self.params[param_choice]
        self.time_step = time_step
        self.last_cell_states = last_cell_states

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
            if  i+self.time_step+self.last_cell_states < len:
                for_input = np.reshape(close[i:i+self.time_step],(1,self.time_step))
                inputs.append(np.matrix.tolist(for_input))
                i = i+self.time_step
                for_targets = np.reshape(close[i:i+self.last_cell_states],(1,self.last_cell_states))
                targets.append(np.matrix.tolist(for_targets))
            else:    
                break
        #     for_input = close[]

        # #number of data elements
        # len = close.shape[0]

        # #sequence size
        # sequence = self.time_step

        # #finding how many can we make
        # row = int(len/sequence)

        # #number of rows to remove
        # remainder = len - (sequence * row)

        #spliting to have for input and target
        # close = np.split(close,2)

        # #now deleting the first 'remainder' number of rows
        # close = np.delete(close, np.s_[:remainder]).transpose()



        # close = np.reshape(close,(row,self.time_step))
        inputs = np.reshape(inputs,(np.shape(inputs)[0],np.shape(inputs)[2]))
        targets = np.reshape(targets,(np.shape(targets)[0],np.shape(targets)[2]))
        
        return inputs,targets