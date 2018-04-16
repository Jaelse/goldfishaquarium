class Configurations:
    
    input_size=1    #number of features
    time_steps=30   #number of time points in a sequence
    lstm_units_per_cell=4     #number of lstm cell per cell
    num_layers=1    #number of layer
    batch_size=60   #batch size
    init_learning_rate=0.001     #learning rate    
    learning_rate_decay=0.99 #learning rate decay
    max_epoch=1000      #maximum iterations for training

    def __init__(self,input_size=1,time_steps=30,num_layers=1,lstm_units_per_cell=[10],batch_size=60,init_learning_rate=0.001,learning_rate_decay=0.99,max_epoch=1000):
        self.input_size = input_size
        self.time_steps =  time_steps
        self.num_layers = num_layers
        self.lstm_units_per_cell = lstm_units_per_cell
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.max_epoch = max_epoch
    