import numpy as np
import tensorflow as tf
import stock as st
import rnn_configuration as configurations

class StoPreC:
    tf.reset_default_graph
    brain = None
    
    Weights = None
    biases = None

    #TODO make it that it can be initiated from outside
    Data = None

    #configurations
    config = None
    

    targets = None

    #inputs
    inputs = tf.placeholder(tf.float32, [None,60,1])

    #target
    targets = tf.placeholder(tf.float32, [None,1])

    #train_op
    train_op = None

    #loss_op
    loss_op = None

    #accuracy
    accuracy = None

    #units -> number of units this cell
    def lstm_cell(self,units):
        return tf.nn.rnn_cell.LSTMCell(units)


    def __init__(self,config,Data):

        #dataset
        self.Data = Data
        
        self.config = config

        self.Weights = tf.Variable(tf.random_normal([config.lstm_units_per_cell[len(config.lstm_units_per_cell)-1],config.input_size]))
        self.biases = tf.Variable(tf.random_normal([config.lstm_units_per_cell[len(config.lstm_units_per_cell)-1]]))

        #input
        self.inputs = tf.placeholder(tf.float32, [None,config.time_steps,config.input_size])

        #target
        self.targets = tf.placeholder(tf.float32, [None,config.input_size])     

        self.brain = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(config.lstm_units_per_cell[i]) for i in range(config.num_layers)], state_is_tuple=True) if config.num_layers > 1 else lstm_cell(units[0])

        #train
        self.train()

    def brain_work(self,x,weights,biases):
        # Get lstm cell output
        outputs, _ = tf.nn.dynamic_rnn(self.brain, x, dtype=tf.float32)

        #transpose to make it good for multiplication
        outputs = tf.transpose(outputs, [1,0,2])

        #the output from the last layer
        last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="last_lstm_output")

        # Linear activation, using rnn inner loop last output
        return tf.matmul(last, weights) + biases


    def train(self):
        logits = self.brain_work(self.inputs, self.Weights, self.biases)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.targets))
    
        optimizer = tf.train.GradientDescentOptimizer(0.99)
    
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.targets, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def put_it_to_life(self):
        # Start training
        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("/tmp/goldfish_tesorboard/1", sess.graph)
            writer.add_graph(sess.graph)

            # Run the initializer
            sess.run(tf.global_variables_initializer())

        #TODO make batches
            for step in range(1, 10):
                open_st,close_st = self.Data.get_data()
                open_st = np.reshape(open_st,(int(open_st.shape[0]/60),60,1))
                close_st = np.reshape(close_st,(int(close_st.shape[0]/60),60,1))

                batch_x = tf.convert_to_tensor(open_st, tf.float32)
                batch_y = tf.convert_to_tensor(close_st, tf.float32)

                # Run optimization op (backprop)
                sess.run(self.train_op, feed_dict={self.inputs: batch_x, self.targets: batch_y})
                #if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.inputs: batch_x, self.targets: batch_y})
                
                #print("Step " + str(step) + ", Minibatch Loss= " + \ "{:.4f}".format(loss) + ", Training Accuracy= " + \ "{:.3f}".format(acc))

        print("Optimization Finished!")

        # # Calculate accuracy for 128 mnist test images
        # test_len = 128
        # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        # test_label = mnist.test.labels[:test_len]
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))


#Rnn cofigurations
configs = configurations.Configurations(input_size=1,time_steps=30,num_layers=2,lstm_units_per_cell=[10,5],batch_size=60,init_learning_rate=0.001,learning_rate_decay=0.99,max_epoch=1000)

#Dataset
Data = st.Stock(1,1,1)

creature = StoPreC(configs,Data)       

#start the process
creature.put_it_to_life()  