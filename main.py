import numpy as np
import tensorflow as tf
import stock as st
import rnn_configuration as configurations
import matplotlib.pyplot as plt


class LstmRnn(object):
    def __init__(self, configs, data_set):
        self.configs = configs
        self.data_set = data_set
        super(LstmRnn, self).__init__()
        
    def _create_cell(self, units_per_cell, name="didn't name"):
        return tf.nn.rnn_cell.LSTMCell(units_per_cell, state_is_tuple=True, name=name)

    def build_graph(self):
        configs = self.configs

        tf.reset_default_graph()
        lstm_graph = tf.Graph()

        with lstm_graph.as_default():
            inputs = tf.placeholder(tf.float32,
                    [None, configs.time_steps, configs.input_size], name="inputs")
            targets = tf.placeholder(tf.float32,
                    [None, configs.input_size], name="outputs")
            learning_rate = tf.placeholder(tf.float32, None)

            cell = (
                    ( tf.nn.rnn_cell.MultiRNNCell(
                        [
                            self._create_cell(
                                configs.lstm_units_per_cell[i], "layer_"+str(i)
                                ) for i in range(configs.num_layers)
                            ],
                        state_is_tuple=True
                        ) ) if configs.num_layers > 1
                    else ( self._create_cell(configs.lstm_units_per_cell[0], "layer_0") )
                    )
            val, state_ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])

            last =  tf.gather(val, int(val.shape[0]) - 1, name="last_lstm_output")

            # TODO change configs.lstm_units_per_cell for different layers(i.e. cells).
            weight = tf.Variable(tf.truncated_normal(
                [configs.lstm_units_per_cell[0], configs.input_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[configs.input_size]))
            prediction = tf.matmul(last, weight) + bias

            loss = tf.reduce_mean(tf.square(prediction - targets))
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            minimize = optimizer.minimize(loss)

    # def train(self):
    #     configs = self.configs
        with tf.Session(graph = lstm_graph) as sess:
            tf.global_variables_initializer().run()
            learning_rates_to_use = [
                    configs.init_learning_rate * (
                        configs.learning_rate_decay ** max(float(i + 1 - configs.init_epoch), 0.0)
                        ) for i in range(configs.max_epoch)
                    ]
            # batch_X, batch_y = self.data_set.generate_one_epoch(configs.batch_size)
            # print("X:")
            # print(batch_X)
            # print(batch_X.shape)
            # print("y:")
            # print(batch_y)
            # print(batch_y.shape)
            for epoch_step in range(configs.max_epoch):
                current_learning_rate = learning_rates_to_use[epoch_step]
                self.data_set.generate_one_epoch(configs.batch_size)
                for batch_X, batch_y in self.data_set.generate_one_epoch(configs.batch_size):
                    train_data_feed = {
                            inputs: batch_X,
                            targets: batch_y,
                            learning_rate: current_learning_rate
                            }
                    training_loss, _ = sess.run([loss, minimize], train_data_feed)
                    print("training loss: " + str( training_loss ))

if __name__ == '__main__':
    # Rnn cofigurations
    configs = configurations.Configurations(
            input_size=1,
            time_steps=3,
            num_layers=1,
            lstm_units_per_cell=[10],
            batch_size=5,
            init_learning_rate=0.01,
            learning_rate_decay=0.99,
            max_epoch=50
            )
    configs.init_epoch = 5
    data_set = st.Stock(1, 1,
            configs.lstm_units_per_cell[len(configs.lstm_units_per_cell)-1],
            configs)

    X, y = data_set.get_data()

    rnn = LstmRnn(configs, data_set)
    rnn.build_graph()
