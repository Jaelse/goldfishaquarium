import numpy as np
import tensorflow as tf
import stock as st
import rnn_configuration as configurations
from matplotlib import pyplot as plt
import time 

class StoPreC:
    # units -> number of units this cell
    def lstm_cell(self,units,name="brain"):
        lstm = tf.nn.rnn_cell.LSTMCell(units, state_is_tuple=True, name="brain_"+name)
        return lstm

    def __init__(self,config,Data):
        tf.reset_default_graph
        self.graph = tf.Graph()

        self.Weights = []
        self.biases = []

        # dataset
        self.Data = Data
        
        # configurations for this object
        self.config = config

       
    def brain_work(self):

        with self.graph.as_default():
            #between dense layers
            for i in range(self.config.normal_num_layers):
                if i is 0:
                    #first weight from the memory
                    self.Weights.append(
                        tf.Variable(
                            tf.truncated_normal(
                                [self.config.lstm_units, self.config.units_per_layer[i]]
                            ), name="Weight_0"
                        )
                    )

                else:
                    name = "Weight_"+str(i)
                    self.Weights.append(
                        tf.Variable(
                            tf.truncated_normal(
                                [self.config.units_per_layer[i-1], self.config.units_per_layer[i]]
                                ), name=name
                            )
                        ) 

            # biases of every layer
            for i in range(self.config.normal_num_layers):
                name = "biases_"+str(i)
                self.biases.append(
                    tf.Variable(
                        tf.constant(0.1, shape=[self.config.units_per_layer[i]])
                    , name=name)
                )

            # input
            self.inputs = tf.placeholder(tf.float32, [None,self.config.time_steps,self.config.input_size],name="Inputs")

            # target
            targets = tf.placeholder(tf.float32, [None,self.config.input_size],name="Targets")     

            #putting lstm
            self.memory = (tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(self.config.lstm_units, "lstm"+str(i)) 
                for i in range(self.config.lstm_cells)],state_is_tuple=True) 
                    if self.config.lstm_cells > 1 else self.lstm_cell(self.config.lstm_units, "lstm"+"0"))

            #------------ memory work---------------
            # Get lstm cell output
            outputs, _ = tf.nn.dynamic_rnn(self.memory, self.inputs, dtype=tf.float32)

            # before transpose shape (batch_size, num_steps, lstm_size)
            # transpose to get (num_steps, batch_size, lstm_size)
            outputs = tf.transpose(outputs, [1,0,2])

            #the output from the last layer
            last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="last_lstm_output")

            # -----------dense work--------- 
            for layers in range(self.config.normal_num_layers):
                if layers is 0:
                    logits = tf.matmul(last, self.Weights[layers]) + self.biases[layers]
                else:    
                    logits = tf.matmul(logits, self.Weights[layers]) + self.biases[layers]

            self.logits = logits

            #  ----------- train ------------
            # Define loss and optimizer
            self.loss_op = tf.reduce_mean(tf.square(logits - targets))
            
            optimizer = tf.train.RMSPropOptimizer(0.01)
        
            self.train_op = optimizer.minimize(self.loss_op)


        with tf.Session(graph = self.graph) as sess:  
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("/tmp/goldfish_tesorboard/2", sess.graph)
            writer.add_graph(sess.graph)

            # Run the initializer
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

            inputs_st,targets_st,hh,jjjj = self.Data.get_data()


            # train_inputs = inputs_st[:np.shape(inputs_st)[0]-self.config.time_steps,:,:]
            # train_targets = targets_st[:np.shape(targets_st)[0]-self.config.time_steps,:]

            # print(train_inputs)
            # test_inputs = inputs_st[np.shape(inputs_st)[0]-self.config.time_steps:,:,:]
            # test_targets = targets_st[np.shape(targets_st)[0]-self.config.time_steps:,:]

            # train_inputs = tf.nn.l2_normalize(inputs_st)
            # train_targets = tf.nn.l2_normalize(targets_st)
            

            # TODO make batches
            if self.config.type == "train":
                for step in range(1, 100):

                    try:
                        batch_x = tf.convert_to_tensor(inputs_st, tf.float64)
                        batch_y = tf.convert_to_tensor(targets_st, tf.float64)

                        batch_x = tf.reshape(batch_x, (tf.shape(batch_x)[0],tf.shape(batch_x)[1], self.config.input_size))    

                        # Run optimization op (backprop)
                        # sess.run(self.train_op, feed_dict={inputs: batch_x.eval(), targets: batch_y.eval()})
                        
                        # Calculate batch loss and accuracy
                        loss, _,  pred= sess.run([self.loss_op, self.train_op, logits],
                                            feed_dict={self.inputs: batch_x.eval(),
                                                        targets: batch_y.eval()})
                        
                        print("Step " + str(step) 
                            + ", Batch Loss= " + str(loss))          

                    except KeyboardInterrupt:
                        quit_option = str(input("quit?(y/n): "))
                        if quit_option == "y":
                            break
                        plt.plot(range(len(self.Data.train_y)), self.Data.train_y*self.Data.normFactorX, 'b-')
                        plt.plot(range(len(self.Data.train_y)), pred*self.Data.normFactorX, 'r-')
                        plt.show()

                plt.plot(range(len(self.Data.train_y)), self.Data.train_y*self.Data.normFactorX, 'b-')
                plt.plot(range(len(self.Data.train_y)), pred*self.Data.normFactorX, 'r-')
                plt.show()
                # test_loss, prediction = sess.run([self.loss_op, self.logits], feed_dict={ self.inputs: test_inputs, self.targets: test_targets})
                # print(" Targets= " + str(test_targets) + 
                #         ", Predicteds=" + str(prediction))

                tm  = time.time()
                model_tm = "./models/goldfish.ckpt"
                save_path = saver.save(sess, model_tm)
                
                print("Optimization Finished!")

            elif self.config.type=="get_logits":
                
                logits = None
            
                saver.restore(sess, "./models/goldfish.ckpt")
                ins = self.Data.get_current_data()
                logits = sess.run(self.logits, feed_dict={self.inputs: ins})
                
        
                return ins , logits

    def get_logits(self):
        # inputs = tf.placeholder(tf.float32, [None,self.config.time_steps,self.config.input_size],name="Inputs")

        with tf.Session(graph = self.graph) as sess:  
            sess.run(tf.global_variables_initializer())

            ins = self.Data.get_current_data()
            logits = sess.run(self.logits, feed_dict={self.inputs: ins})

        return ins,logits
if __name__ == '__main__':
    # Rnn cofigurations
    configs = configurations.Configurations(
        types="train",
        input_size=1,
        time_steps=60,
        normal_num_layers=4,
        units_per_layer=[60, 30, 1], 
        lstm_cells = 2,
        lstm_units = 124,
        batch_size=10,
        init_learning_rate=0.001,
        learning_rate_decay=0.99,
        max_epoch=1000,
        keep_prob=0.8)

    # Dataset
    data = st.Stock(0,3,0,configs)
    creature = StoPreC(configs, data)

    # start the process
    creature.brain_work()

