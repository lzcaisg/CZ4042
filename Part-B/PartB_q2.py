import matplotlib as mpl
mpl.use('Agg')

import os
import logging
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from nn_models import placehoder_inputs, hidden_layers, output_layer
from utils import get_data, split_train_test, get_batch_data

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
NUM_FEATURES = 8

LEARNING_RATES = [0.5e-6, 1e-7, 0.5e-8, 1e-9, 1e-10]
NUM_NEURONS=[30]
BATCH_SIZE = 256
EPOCHS = 2000
BETA = 1e-3
NO_FOLDS = 5

SEED = 10
np.random.seed(SEED)

IMAGE_DIR = 'image'
if not os.path.isdir(IMAGE_DIR):
    logger.warn("image folder not exists, creating folder")
    os.makedirs(IMAGE_DIR)


if __name__ == "__main__":
    # Load dataset
    X_data, y_data = get_data('cal_housing.data')
    # Split the dataset into validation and test dataset
    # Validation dataset 70%
    # Test dataset 30%
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=SEED)
    logger.info('Train shape: {}, Test shape: {}, ratio: {}'.format(
        X_train.shape, X_test.shape, X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]))
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train / 1000
    y_test = y_test / 1000

    # Create neural networks
    x, y_ = placehoder_inputs(NUM_FEATURES, 1)
    h = hidden_layers(neurons=NUM_NEURONS, input_tensor=x)
    y = output_layer(h)

    # L2 Regularization
    regularizer = tf.contrib.layers.l2_regularizer(scale=BETA)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    # Loss
    loss = tf.reduce_mean(tf.square(y_ - y)) + reg_term
    # Error 
    error = tf.reduce_mean(tf.square(y_ - y))
    # Create the gradient descent optimizer with the learning rate

    lr_error_mean = []
    '''
    for lr in LEARNING_RATES:
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss)

        kf = KFold(n_splits=NO_FOLDS)
        errors = []
        fold = 1
        for train_index, test_index in kf.split(X_train):
            logger.info("Learning_rate: {}, fold {}".format(lr, fold))
            fold += 1
            X, X_val = X_train[train_index], X_train[test_index]
            y, y_val = y_train[train_index], y_train[test_index]

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # for i in tqdm(range(epochs)):
                for _ in tqdm(range(EPOCHS)):
                    data_gen = get_batch_data(X, y, BATCH_SIZE)
                    for X_batch, y_batch in data_gen:
                        sess.run(train_op, feed_dict={x: X_batch, y_: y_batch})
                
                err = error.eval(feed_dict={x: X_val, y_: y_val})
                errors.append(err)
        mean_error = sum(errors) / float(len(errors))
        lr_error_mean.append(mean_error)
        print(mean_error)

    plt.figure()
    x = range(len(LEARNING_RATES))
    plt.plot(x, lr_error_mean)
    lrticks = [str(_) for _ in LEARNING_RATES]
    plt.xticks(x, lrticks)
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Cross Validation Error')
    plt.title('Learning Rate')
    plt.savefig(os.path.join(IMAGE_DIR, "PartB_q2a"))
    '''

    optimal_lr = 0.5e-6
    lr_test_errors = []
    for lr in LEARNING_RATES:
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_error_epoch = []

            for i in range(EPOCHS):
                # Get a banch of data with batch_size = 32
                data_gen = get_batch_data(X_train, y_train, BATCH_SIZE)
                # initialize train_error for each epoch
                for X_batch, y_batch in data_gen:
                    train_op.run(feed_dict={x: X_batch, y_: y_batch})
                
                # Evaluation test error
                test_error = error.eval(feed_dict={x: X_test, y_: y_test})
                test_error_epoch.append(test_error)

                if i % 100 == 0:
                    logger.info('epoch:{}, testing error:{}'.format(i, test_error_epoch[i]))
            

            lr_test_errors.append(test_error_epoch)
            if lr == optimal_lr:
                logger.info("Plotting test errors")
                plt.figure(1)
                plt.plot(range(EPOCHS), test_error_epoch)
                plt.xlabel(str(EPOCHS) + ' iterations')
                plt.ylabel('Test Error')
                plt.savefig(os.path.join(IMAGE_DIR, "PartB_q2b"))


    plt.figure(2)
    for test_errors in lr_test_errors:
        plt.plot(range(EPOCHS), test_errors)

    plt.legend(["lr = {}".format(lr) for lr in LEARNING_RATES], loc='upper right')
    plt.xlabel(str(EPOCHS) + 'iterations')
    plt.ylabel('Test Error')
    plt.savefig(os.path.join(IMAGE_DIR, "PartB_q2c"))

