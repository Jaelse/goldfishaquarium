class Configurations:

    def __init__(self,types="train",input_size=1,time_steps=30,normal_num_layers=1,units_per_layer=10,lstm_cells=1,lstm_units=10,batch_size=60,init_learning_rate=0.001,learning_rate_decay=0.99,max_epoch=1000,keep_prob=1):
        self.type = types
        self.input_size = input_size
        self.time_steps =  time_steps
        self.normal_num_layers = normal_num_layers
        self.units_per_layer = units_per_layer
        self.lstm_cells =  lstm_cells
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.max_epoch = max_epoch
        self.keep_prob = keep_prob
    
