import matplotlib as mpl
mpl.use('Agg')

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from nn_models import placeholder_inputs, hidden_layers, output_layer
from utils import get_data, get_batch_data

os.environ["CUDA_VISIBLE_DEVICES"]="0"

NUM_FEATURES = 8
BATCH_SIZE = 32
EPOCHS = 20000
LEARNING_RATE = 1e-9
KEEP_PROB = 0.9
HIDDEN_LAYERS = [[60], [60, 20], [60, 20, 20]]

logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('tmp/q4.log')
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

# Seed
SEED = 10
np.random.seed(SEED)

IMAGE_DIR = 'image'
if not os.path.isdir(IMAGE_DIR):
    logger.warn("image folder not exists, creating folder")
    os.makedirs(IMAGE_DIR)


def train(keep_prob=None):
    model_test_errors = []

    for h_layers in HIDDEN_LAYERS:
        if keep_prob is None:
            logger.info("Training {} layer model".format(len(h_layers)+2))
            log_dir = "tmp/partb_4/{}_layers".format(len(h_layers)+2)
        else:
            logger.info("Training {} layer model with dropout".format(len(h_layers)+2))
            log_dir = "tmp/partb_4/{}_layers_dropout".format(len(h_layers)+2)
        graph = tf.Graph()
        with graph.as_default():
            x, y_ = placeholder_inputs(NUM_FEATURES, 1)
            h = hidden_layers(neurons=h_layers, input_tensor=x, keep_prob=keep_prob)
            y = output_layer(input_tensor=h, output_dim=1)

            loss = tf.reduce_mean(tf.square(y_ - y))
            error = tf.reduce_mean(tf.square(y_ - y))
            optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            train_op = optimizer.minimize(loss)
        
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(log_dir)
            writer.add_graph(sess.graph)

            test_error_epoch = []
            for i in tqdm(range(EPOCHS)):
                data_gen = get_batch_data(X_train, y_train, BATCH_SIZE)
                for X_batch, y_batch in data_gen:
                    train_op.run(feed_dict={x: X_batch, y_: y_batch})

                test_error = error.eval(feed_dict={x: X_test, y_: y_test})
                test_error_epoch.append(test_error)
                # if i % 500 == 0:
                #    logger.info('iter %d: test error %g' % (i,  test_error_epoch[i]))

            model_test_errors.append(test_error_epoch)

    logger.info("Plotting Image")
    plt.figure()
    for test_errors in model_test_errors:
        plt.plot(range(EPOCHS), test_errors)
    plt.legend(["{} layers".format(len(h_layers)+2) for h_layers in HIDDEN_LAYERS], loc="upper right")
    plt.xlabel('Epochs')
    plt.ylabel('Test Error')
    if keep_prob is None:
        title = "Different Numbers of Hidden Layers"
        name = "PartB_q4a.png"
    else:
        title = "Different Numbers of Hidden Layers with dropout"
        name = "PartB_q4b.png"
    plt.title(title)
    plt.savefig(os.path.join(IMAGE_DIR, name))

if __name__ == '__main__':
       # prepare data
    X_data, y_data = get_data('cal_housing.data')
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=SEED)

    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train / 1000
    y_test= y_test / 1000


    train()
    # logger.info("Train {} layer model with dropout".format(index))
