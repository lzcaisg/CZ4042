import os
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nn_models import placehoder_inputs, hidden_layers, output_layer

# Hyperparameters
NUM_FEATURES = 8

batch_size = 32
learning_rate = 1e-7
epochs = 500
beta = 1e-3
num_neuron = 30

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
SEED = 10
np.random.seed(SEED)

IMAGE_DIR = 'image'
if not os.path.isdir(IMAGE_DIR):
    logger.warn("image folder not exists, creating folder")
    os.makedirs(IMAGE_DIR)


def get_batch_data(X_train, y_train, batch_size):
    idx = np.arange(len(X_train))
    N = len(X_train)
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]

    for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
        yield X_train[start: end], y_train[start: end]


def get_data():
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
    X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
    Y_data = (np.asmatrix(Y_data)).transpose()
    return X_data, Y_data


def split_train_test(X_data, Y_data, test_ratio=0.30):
    '''Split the validating and testing datasets
        test_radio (float, optional): Defaults to 0.30. ratio of test datasets
    '''
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_ratio, random_state=SEED)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Prepare validaing and testing datasets
    logger.info('Train Test Split')
    X_data, y_data = get_data()
    X_train, X_test, y_train, y_test = split_train_test(X_data, y_data, 0.3)
    logger.info('Train shape: {}, Test shape: {}, ratio: {}'.format(
        X_train.shape, X_test.shape, X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]))
    )

    # Create neural networks
    x, y_ = placehoder_inputs(NUM_FEATURES, 1)
    h = hidden_layers(neurons=[30], input_tensor=x)
    y = output_layer(h)

    # L2 Regularization
    regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

    # Loss
    loss = tf.reduce_mean(tf.square(y_ - y)) + reg_term

    # Error 
    error = tf.reduce_mean(tf.square(y_ - y))

    # Create the gradient descent optimizer with the learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    # saver = tf.train.Saver()

    # Train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        N = len(X_train)
        idx = np.arange(N)
        train_error_epoch = []

        writer = tf.summary.FileWriter("tmp/part2_models/q1")
        writer.add_graph(sess.graph)
        logger.info("model saved")

        test_error_epoch = []
        for i in tqdm(range(epochs)):
            # Start of the epoch
            # Shuffle data
            data_gen = get_batch_data(X_train, y_train, batch_size)
            train_error = 0
            for X_batch, y_batch in data_gen:
                train_op.run(feed_dict={x: X_batch, y_: y_batch})
                train_error += error.eval(feed_dict={x: X_batch, y_: y_batch})
            

        test_idx = np.arange(len(X_test))
        sample_idx = np.random.choice(test_idx, 50)
        X_sample, y_sample = X_test[sample_idx], y_test[sample_idx]
        y_pred = y.eval(feed_dict={x: X_sample})

        prediction = y_pred.flatten()
        target = np.asarray(y_sample).reshape(-1)

        print(prediction, prediction.shape)
        print(target, target.shape)

        logger.info("Plotting training errors")
        plt.figure(1)
        plt.plot(range(epochs), train_error_epoch)
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.savefig(os.path.join(IMAGE_DIR, "1. Training Error.png"))
        logger.info("image saved")


        plt.figure(2)
        plt.scatter(target, prediction)
        plt.plot(target, target, color='g')
        plt.xlabel('$prediction$')
        plt.ylabel('$target$')
        plt.title('Prediction Target Plot')
        plt.savefig(os.path.join(IMAGE_DIR, "1. Prediction vs Target Plot.png"))


