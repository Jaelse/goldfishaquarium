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
        units_per_layer=[60, 1], 
        lstm_cells = 1,
        lstm_units = 120,
        batch_size=10,
        init_learning_rate=0.0001,
        learning_rate_decay=0.99,
        max_epoch=1000,
        keep_prob=0.8)

        # Dataset
        self.data = stock.Stock(0,3,0,self.configs)

        self.creature = stop.StoPreC(self.configs, self.data)


        style.use('fivethirtyeight')
        self.fig = plt.figure()

        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        # self.ax1 = self.fig.add_subplot(221)
        # self.ax2 = self.fig.add_subplot(222)
        # self.ax3 = self.fig.add_subplot(223)
        # self.ax4 = self.fig.add_subplot(224)

        x,logit = self.creature.brain_work()
        self.xs = np.reshape(x, (-1))
        self.xs = [self.xs[np.shape(self.xs)[0]-1] *self.data.normFactor ]
        self.ys = [logit[0,0] *self.data.normFactor ]

    def animate_train(self,i):
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.set_title("Real Values")
        self.ax1.plot(range(np.shape(self.xs)[0]),self.xs)
        self.ax1.annotate(str(self.xs[-1]),
                xy=(np.shape(self.xs)[0]-1, self.xs[-1]),  xycoords='data',
                    xytext=(0.8, 0.95), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )

        self.ax2.set_title("Perdicted Values")
        self.ax2.plot(range(1,np.shape(self.ys)[0]+1),self.ys)
        self.ax2.annotate(str(self.ys[-1]),
                xy=(np.shape(self.ys)[0], self.ys[-1]),  xycoords='data',
                    xytext=(0.8, 0.95), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='right', verticalalignment='top',
                    )

        print("input_last:" + str(self.xs[np.shape(self.xs)[0]-1]))
        print("output_last:" + str(self.ys[np.shape(self.ys)[0]-1]))

        xs,logits = self.creature.get_logits()

        new_xs = np.reshape(xs, (-1))[self.configs.time_steps-1]
        self.xs = np.append(self.xs,[new_xs *self.data.normFactor ])
        
        self.ys = np.append(self.ys, [logits *self.data.normFactor ])
        
        
    def show_training(self):
        anim = animation.FuncAnimation(self.fig, self.animate_train, interval=1000*60)
        
        plt.show()

dv = DataVisualization()
dv.show_training()
