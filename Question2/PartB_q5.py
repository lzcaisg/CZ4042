from nn_models import placehoder_inputs, hidden_layers, output_layer
from PartB_q1 import get_data, split_train_test, get_batch_data
import tensorflow as tf
import logging
NUM_FEATURES = 8

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(graph, neurons, learning_rate=1e-9, keep_prob=None):
    with graph.as_default():
        x, y_ = placehoder_inputs(NUM_FEATURES, 1)
        h = hidden_layers(neurons=neurons, input_tensor=x, keep_prob=keep_prob)
        y = output_layer(input_tensor=h, output_dim=1)

        loss = tf.reduce_mean(tf.square(y_ - y))
        error = tf.reduce_mean(tf.square(y_ - y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        return error, train_op

def train_model(graph, batch_size, train_op, error, log_dir):
    # prepare data
    X_data, y_data = get_data()
    X_train, X_test, y_train, y_test = split_train_test(X_data, y_data, 0.3)
    
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(sess.graph)
        data_gen = get_batch_data(X_train, y_train, batch_size)

        '''
        for i in tqdm(range(epochs)):
            for X_batch, y_batch in data_gen:
                train_op.run(feed_dict={x: X_batch, y_: y_batch})

            if i % 100 == 0:
                err = error.eval(feed_dict={x: X_test, y_: y_batch})
              	print('iter %d: test error %g' % (i, err))
        '''

def train_model_without_dropout():
    # define graphs
    graph_4_layers = tf.Graph()
    graph_5_layers = tf.Graph()
    graph_3_layers = tf.get_default_graph()

    graphs = [graph_3_layers, graph_4_layers, graph_5_layers]
    neurons = [[20], [20, 20], [20, 20, 20]]

    for index, (graph, neuron) in enumerate(zip(graphs, neurons), 3):
        logger.info("Create {} layer model".format(index))
        error, train_op = create_model(graph, neuron)
        logger.info("Train {} layer model".format(index))
        train_model(graph, 32, train_op, error, "tmp/partb_4/{}_layers".format(index))

def train_model_with_dropout(keep_prob):
    graph_4_layers = tf.Graph()
    graph_5_layers = tf.Graph()
    graph_3_layers = tf.get_default_graph()

    graphs = [graph_3_layers, graph_4_layers, graph_5_layers]
    neurons = [[20], [20, 20], [20, 20, 20]]

    for index, (graph, neuron) in enumerate(zip(graphs, neurons), 3):
        logger.info("Create {} layer model with dropout".format(index))
        error, train_op = create_model(graph, neuron, keep_prob=keep_prob)
        logger.info("Train {} layer model with dropout".format(index))
        train_model(graph, 32, train_op, error, "tmp/partb_4/{}_layers with dropout".format(index))

if __name__ == '__main__':
    train_model_with_dropout(0.9)
    train_model_without_dropout()