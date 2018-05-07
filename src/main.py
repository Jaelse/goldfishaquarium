import numpy as np
import tensorflow as tf
import stock as st
import rnn_configuration as configurations
import matplotlib.pyplot as plt


class LstmRnn(object):
    def __init__(self, configs, data_set, model_path="../model/model"):
        super(LstmRnn, self).__init__()

        self.configs = configs
        self.data_set = data_set
        self.model_path = model_path 
        
    def _create_cell(self, units_per_cell, name="didn't name"):
        return tf.nn.rnn_cell.LSTMCell(units_per_cell, state_is_tuple=True, name=name)

    def build_graph(self):
        configs = self.configs

        tf.reset_default_graph()
        self.lstm_graph = tf.Graph()

        with self.lstm_graph.as_default():
            self.inputs = tf.placeholder(tf.float32,
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
            val, state_ = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])

            last =  tf.gather(val, int(val.shape[0]) - 1, name="last_lstm_output")

            # TODO change configs.lstm_units_per_cell for different layers(i.e. cells).
            weight = tf.Variable(tf.truncated_normal(
                [configs.lstm_units_per_cell[0], configs.input_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[configs.input_size]))
            self.prediction = tf.matmul(last, weight) + bias

            # loss = tf.reduce_mean(tf.square(self.prediction - targets))
            loss = tf.losses.mean_squared_error(targets, self.prediction)
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            minimize = optimizer.minimize(loss)
            merged_summary = tf.summary.merge_all()

        with tf.Session(graph = self.lstm_graph) as sess:
            tf.global_variables_initializer().run()

            writer = tf.summary.FileWriter("/tmp/goldfish_tesorboard/2", sess.graph)
            writer.add_graph(sess.graph)

            test_data_feed = {
                    self.inputs: self.data_set.test_X,
                    targets: self.data_set.test_y,
                    learning_rate: 0.0
                    }
            print(self.data_set.test_X.shape)

            learning_rates_to_use = [
                    configs.init_learning_rate * (
                        configs.learning_rate_decay ** max(float(i + 1 - configs.init_epoch), 0.0)
                        ) for i in range(configs.max_epoch)
                    ]
            for epoch_step in range(configs.max_epoch):
                current_learning_rate = learning_rates_to_use[epoch_step]
                self.data_set.generate_one_epoch(configs.batch_size)

                try:
                    # Train
                    for batch_X, batch_y in self.data_set.generate_one_epoch(configs.batch_size):
                        train_data_feed = {
                                self.inputs: batch_X,
                                targets: batch_y,
                                learning_rate: current_learning_rate
                                }
                        training_loss, _ = sess.run([loss, minimize], train_data_feed)

                    # Test
                    # print(self.data_set.test_y.shape)
                    
                    test_loss, _pred = sess.run([loss, self.prediction], test_data_feed)

                    # Print Train and Test
                    print("final prediction: "+str(_pred.shape))
                    # print(np.concatenate((_pred, self.data_set.test_y), axis=1))
                    print( "epoch: %d | training loss: %f | test loss: %f" % ( epoch_step, training_loss, test_loss ) )
                except KeyboardInterrupt:
                    print("KeyboardInterrupt")
                    quit_option = str( input("quit?(y/n): ") )
                    if quit_option.lower() == 'y':
                        break;
                    print(self.data_set.test_y.shape)
                    print(_pred.shape)
                    predConcat = np.concatenate(  _pred*self.data_set.normFactory )
                    actualConcat = np.concatenate( self.data_set.test_y*self.data_set.normFactory )
                    plt.plot(range(len(predConcat)), predConcat, 'r-')
                    plt.plot(range(len(actualConcat)), actualConcat, 'b-')
                    plt.show()

            # print(self.data_set.test_y.shape)
            # print(_pred.shape)
            predConcat = np.concatenate(  _pred*self.data_set.normFactory )
            actualConcat = np.concatenate( self.data_set.test_y*self.data_set.normFactory )
            plt.plot(range(len(predConcat)), predConcat, 'r-')
            plt.plot(range(len(actualConcat)), actualConcat, 'b-')
            plt.show()

            print("Saving model to: "+self.model_path)
            saver = tf.train.Saver()
            saver.save(sess, self.model_path, global_step=configs.max_epoch)

    def predict(self, input_X):
        with tf.Session(graph = self.lstm_graph) as sess:
            tf.global_variables_initializer().run()

            data_feed = {
                    self.inputs: input_X,
                    }

            _pred = sess.run([self.prediction], data_feed)

            # print(self.data_set.test_y.shape)
            predConcat = _pred[0]*self.data_set.normFactory
            normalizedInput = input_X[-1]*self.data_set.normFactory
            print(len( normalizedInput ))
            print(len( predConcat ))
            plt.plot(range(50+len(predConcat)), predConcat, 'r-')
            plt.plot(range(len(normalizedInput)), normalizedInput, 'b-')
            plt.show()

if __name__ == '__main__':
    # Rnn cofigurations
    # configs = configurations.Configurations(
    #         input_size=5,
    #         time_steps=30,
    #         num_layers=2,
    #         lstm_units_per_cell=[128, 128],
    #         batch_size=64,
    #         init_learning_rate=0.001,
    #         learning_rate_decay=0.99,
    #         max_epoch=200
    #         )
    configs = configurations.Configurations(
            input_size=1,
            time_steps=50,
            num_layers=1,
            lstm_units_per_cell=[128],
            batch_size=64,
            init_learning_rate=0.0001,
            learning_rate_decay=0.8,
            max_epoch=50
            )
    configs.init_epoch = 5

    params = 0 # 0Dow Jones, 1Apple, 2Facebook
    intervals = 3 # 0sec, 1minute, 2hour, 3day, 4month.
    period = 0 # 0year, 1month, 2day

    data_set = st.Stock(params, intervals, period, configs)

    # data_set.get_data()
    # print(data_set.get_current_data())

    rnn = LstmRnn(configs, data_set)
    rnn.build_graph()

    input_X = data_set.other_input
    rnn.predict(input_X)
