import numpy as np
import tensorflow as tf
import stock as st
import rnn_configuration as configurations
from matplotlib import pyplot as plt

class StoPreC:
    tf.reset_default_graph

    # units -> number of units this cell
    def lstm_cell(self,units,name="brain"):
        lstm = tf.nn.rnn_cell.LSTMCell(units,name="brain_"+name)
        return lstm

    def __init__(self,config,Data):
        self.Weights = []
        self.biases = []

        # dataset
        self.Data = Data
        
        # configurations for this object
        self.config = config

        #first weight from the memory
        self.Weights.append(
            tf.Variable(
                tf.truncated_normal(
                    [config.lstm_units, config.units_per_layer[0]]
                ), name="Weight_0"
            )
        )
        #between dense layers
        for i in range(config.normal_num_layers-1):
            name = "Weight_"+str(i)
            self.Weights.append(
                tf.Variable(
                    tf.truncated_normal(
                        [config.units_per_layer[i], config.units_per_layer[i+1]]
                        ), name=name
                    )
                ) 

        # biases of every layer
        for i in range(config.normal_num_layers):
            name = "biases_"+str(i)
            self.biases.append(
                tf.Variable(
                    tf.constant(0.1, shape=[config.units_per_layer[i]])
                , name=name)
            )

        # input
        self.inputs = tf.placeholder(tf.float32, [None,config.time_steps,config.input_size],name="Inputs")

        # target
        self.targets = tf.placeholder(tf.float32, [None,config.input_size],name="Targets")     

        #putting lstm
        self.memory = (tf.nn.rnn_cell.MultiRNNCell(
            [self.lstm_cell(config.lstm_units, "lstm"+str(i)) 
            for i in range(config.lstm_cells)],state_is_tuple=True) 
                if config.lstm_cells > 1 else self.lstm_cell(config.lstm_units, "lstm"+"0"))

        self.train()

    def memory_work(self, x):
        # Get lstm cell output
        outputs, _ = tf.nn.dynamic_rnn(self.memory, x, dtype=tf.float32)

        # before transpose shape (batch_size, num_steps, lstm_size)
        # transpose to get (num_steps, batch_size, lstm_size)
        outputs = tf.transpose(outputs, [1,0,2])

        #the output from the last layer
        last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="last_lstm_output")
        
        #pass the outputs to dense_work
        return tf.matmul(last, self.Weights[0]) + self.biases[0]

    def dense_work(self, lstm_outputs):
        outputs = lstm_outputs
        for layers in range(self.config.normal_num_layers-1):
            outputs = tf.matmul(outputs, self.Weights[layers+1]) + self.biases[layers+1]

        return outputs    

    def train(self):
        logits = self.dense_work(self.memory_work(self.inputs))
        print(logits)
        self.logits = logits
        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.square(logits - self.targets))
        
        optimizer = tf.train.RMSPropOptimizer(0.01)
    
        self.train_op = optimizer.minimize(self.loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(logits, self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print("\tAccuracy:")
        print("\t\t"+str( tf.metrics.accuracy(self.targets, logits).count ))
        print("............Done Training............")

    def put_it_to_life(self):
        with tf.Session() as sess:  
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("/tmp/goldfish_tesorboard/2", sess.graph)
            writer.add_graph(sess.graph)

            # Run the initializer
            sess.run(tf.global_variables_initializer())

            inputs_st,targets_st = self.Data.get_data()


            # train_inputs = inputs_st[:np.shape(inputs_st)[0]-self.config.time_steps,:,:]
            # train_targets = targets_st[:np.shape(targets_st)[0]-self.config.time_steps,:]

            # print(train_inputs)
            # test_inputs = inputs_st[np.shape(inputs_st)[0]-self.config.time_steps:,:,:]
            # test_targets = targets_st[np.shape(targets_st)[0]-self.config.time_steps:,:]

            train_inputs = tf.nn.l2_normalize(inputs_st)
            train_targets = tf.nn.l2_normalize(targets_st)
            

            # TODO make batches
            for step in range(1, 400):

                batch_x = tf.convert_to_tensor(inputs_st, tf.float64)
                batch_y = tf.convert_to_tensor(targets_st, tf.float64)

                batch_x = tf.reshape(batch_x, (tf.shape(batch_x)[0],tf.shape(batch_x)[1], self.config.input_size))    

                # Run optimization op (backprop)
                sess.run(self.train_op, feed_dict={self.inputs: batch_x.eval(), self.targets: batch_y.eval()})
                
                # Calculate batch loss and accuracy
                loss, acc, _pred= sess.run([self.loss_op, self.accuracy, self.logits],
                                     feed_dict={self.inputs: batch_x.eval(),
                                                self.targets: batch_y.eval()})
                
                print("Step " + str(step) 
                    + ", Batch Loss= " + "{:.4f}".format(loss))          

            predConcat = np.concatenate(  _pred )
            actualConcat = np.concatenate( self.Data.targets )
            plt.plot(range(len(predConcat)), predConcat, 'r-')
            plt.plot(range(len(actualConcat)), actualConcat, 'b-')
            plt.show()

            # test_loss, prediction = sess.run([self.loss_op, self.logits], feed_dict={ self.inputs: test_inputs, self.targets: test_targets})
            # print(" Targets= " + str(test_targets) + 
            #         ", Predicteds=" + str(prediction))
        print("Optimization Finished!")

# Rnn cofigurations
configs = configurations.Configurations(
    input_size=1,
    time_steps=30,
    normal_num_layers=2,
    units_per_layer=[50,1],
    lstm_cells = 2,
    lstm_units = 100,
    batch_size=10,
    init_learning_rate=0.001,
    learning_rate_decay=0.99,
    max_epoch=1000,
    keep_prob=0.8)

# Dataset
data = st.Stock(0,0,1,configs)

creature = StoPreC(configs, data)

# start the process
creature.put_it_to_life()
