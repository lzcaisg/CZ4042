from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
from sat_model import *

FLAGS = None
NUM_CLASSES = 6

seed = 10
np.random.seed(seed)

train_file = 'sat_train.txt'
test_file = 'sat_train.txt'

def placehoder_inputs(batch_size):
    features_placeholder = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    labels_placeholder = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    return features_placeholder, labels_placeholder

def main(_):
    trainX, trainY = process_inputs(train_file)
    testX, testY = process_inputs(test_file)

    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

    batch_size = FLAGS.batch_size
    hidden1 = FLAGS.hidden1
    hidden2 = FLAGS.hidden2
    learning_rate = FLAGS.learning_rate
    epochs = FLAGS.epochs
    beta = FLAGS.beta

    with tf.Session() as sess:
        x, y_ = placehoder_inputs(batch_size)

        y, loss = inference(x, y_, hidden1, beta)
        
        train_op = training(loss, learning_rate)

        accuracy = evaluation(y, y_)

        sess.run(tf.global_variables_initializer())

        test_acc = []
        N = len(trainX)
        idx = np.arange(N)
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX = trainX[idx]
            trainY = trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start: end], y_: trainY[start: end]})
            
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            if i % 100 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc[i]))

    # plot learning curves
    plt.figure(1)
    plt.plot(range(epochs), test_acc)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Train accuracy')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=10e-6,
        help='beta for L2 regulation'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Epochs'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=10,
        help='Number of units in hidden layer 1'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=10,
        help='Number of units in hidden layer 2'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    