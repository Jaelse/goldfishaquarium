import rnn_configuration as configuration
import stock as stock
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import animation as animation
from matplotlib import style 
import tensorflow as tf
import stoprec as stop
import numpy as np

class DataVisualization:

    def __init__(self):
        self.configs = configuration.Configurations(
            types="get_logits",
            input_size=1,
            time_steps=60,
            normal_num_layers=2,
            units_per_layer=[30,1], 
            lstm_cells = 1,
            lstm_units = 124,
            batch_size=10,
            init_learning_rate=0.001,
            learning_rate_decay=0.99,
            max_epoch=1000,
            keep_prob=0.8)

        # Dataset
        self.data = stock.Stock(0,3,0,self.configs)

        self.creature = stop.StoPreC(self.configs, self.data)


        style.use('fivethirtyeight')
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)

        x,logit = self.creature.brain_work()
        self.xs = np.reshape(x, (-1))
        self.ys = np.zeros(np.shape(self.xs)[0])
        self.ys = np.append(self.ys, [logit[0,0]])

    def animate_train(self,i):
        # graph_data = open('stock.txt','r').read()
        # lines = graph_data.split('\n')
        # xs = []
        # ys = []
        # for line in lines:
        #     if len(line) > 1:
        #         x, y = line.split(',')
        #         xs.append(x)
        #         ys.append(y)


        self.ax1.clear()
        self.ax1.plot(range(np.shape(self.ys)[0]),self.ys)
        self.ax1.plot(range(np.shape(self.xs)[0]),self.xs)

        xs,logits = self.creature.get_logits()

        new_xs = np.reshape(xs, (-1))[self.configs.time_steps-1]

        self.xs = np.append(self.xs,[new_xs])
        
        self.ys = np.append(self.ys, [logits])
        
        
    def show_training(self):
        anim = animation.FuncAnimation(self.fig, self.animate_train, interval=1000*60)
        
        plt.show()

dv = DataVisualization()
dv.show_training()