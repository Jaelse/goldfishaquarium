import numpy as np
import tensorflow as tf
import stock as st
import rnn_configuration as configurations
import matplotlib.pyplot as plt

class StoPreC:
    tf.reset_default_graph

    # units -> number of units this cell
    def lstm_cell(self,units,name="brain"):
        return tf.nn.rnn_cell.LSTMCell(units,name="brain_"+name)

    def __init__(self,config,Data):
        # dataset
        self.Data = Data
        
        # configurations for this object
        self.config = config

        self.Weights = tf.Variable(
            tf.random_normal(
                [config.lstm_units_per_cell[len(config.lstm_units_per_cell)-1],
                 config.input_size],
                dtype=tf.float64,
                name="Weights"
            )
        )
        self.biases = tf.Variable(
            tf.random_normal(
                [config.lstm_units_per_cell[len(config.lstm_units_per_cell)-1]],
                dtype=tf.float64
            ),
            name="baises"
        )

        # input
        self.inputs = tf.placeholder(tf.float64, [None,config.time_steps,config.input_size])

        # target
        self.targets = tf.placeholder(tf.float64, [None,config.time_steps,config.input_size])     

        self.brain = (
            tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(config.lstm_units_per_cell[i], str(i)) for i in range(config.num_layers)],
                state_is_tuple=True
            ) if config.num_layers > 1
            else self.lstm_cell(config.lstm_units_per_cell[0], "0")
        )
        # train
        self.train()

    def brain_work(self, x, weights,biases):
        # Get lstm cell output
        outputs, _ = tf.nn.dynamic_rnn(self.brain, x, dtype=tf.float64)
        
        # before transpose shape (time_step,batch_size,input_size)
        # transpose to make it good for multiplication
        outputs = tf.transpose(outputs, [1,0,2])

        #the output from the last layer
        last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name="last_lstm_output")

        # Linear activation, using rnn inner loop last output
        return tf.matmul(last, weights) + biases

    def train(self):
        print("............Started Training............")
        logits = self.brain_work(self.inputs, self.Weights, self.biases)
        prediction = tf.nn.softmax(logits)

        target = tf.reshape(
            self.targets,
            (tf.shape(self.targets)[0], tf.shape(self.targets)[1])
        )

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=target))
    
        optimizer = tf.train.GradientDescentOptimizer(0.99)
    
        self.train_op = optimizer.minimize(self.loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print("\tAccuracy:")
        print("\t\t"+str( tf.metrics.accuracy(target, prediction).count ))
        print("............Done Training............")

    def put_it_to_life(self):
        # Start training
        with tf.Session() as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter("/tmp/goldfish_tesorboard/2", sess.graph)
            writer.add_graph(sess.graph)

            # Run the initializer
            sess.run(tf.global_variables_initializer())

        # TODO make batches
            for step in range(1, 10):
                open_st,close_st = self.Data.get_data()

                batch_x = tf.convert_to_tensor(open_st, tf.float64)
                batch_y = tf.convert_to_tensor(close_st, tf.float64)

                batch_x = tf.reshape(batch_x, (tf.shape(batch_x)[0],tf.shape(batch_x)[1], 1))    
                batch_y = tf.reshape(batch_y, (tf.shape(batch_y)[0],tf.shape(batch_y)[1], 1))    

                # Run optimization op (backprop)
                sess.run(self.train_op, feed_dict={self.inputs: batch_x.eval(), self.targets: batch_y.eval()})
                # if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([self.loss_op, self.accuracy],
                                     feed_dict={self.inputs: batch_x.eval(),
                                                self.targets: batch_y.eval()})
                
                print("Step " + str(step) +
                      ", Minibatch Loss= " + "{:.4f}".format(loss) +
                      ", Training Accuracy= " + "{:.3f}".format(acc))

        print("Optimization Finished!")

        # # Calculate accuracy for 128 mnist test images
        # test_len = 128
        # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        # test_label = mnist.test.labels[:test_len]
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))


# Rnn cofigurations
configs = configurations.Configurations(
    input_size=1,
    time_steps=30,
    num_layers=1,
    lstm_units_per_cell=[30],
    batch_size=60,
    init_learning_rate=0.001,
    learning_rate_decay=0.99,
    max_epoch=1000)

# Dataset
data = st.Stock(1,1,1,configs.time_steps,configs.lstm_units_per_cell[len(configs.lstm_units_per_cell)-1])

inputs, targets = data.get_data()

creature = StoPreC(configs, data)

# start the process
creature.put_it_to_life()
