#
# Project 1, starter code part a
#
import math
import numpy as np
import tensorflow as tf
import pylab as plt

NUM_FEATURES = 36
NUM_CLASSES = 6

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def process_inputs(fileName):
    inputs = np.loadtxt(fileName, delimiter=' ')
    X, _Y = inputs[:, :36], inputs[:, -1].astype(int)
    _Y[_Y == 7] = 6

    Y = np.zeros((_Y.shape[0], NUM_CLASSES))
    Y[np.arange(_Y.shape[0]), _Y - 1] = 1 #one hot matrix
    return X, Y

def inference(x, y_, hidden_units, beta):
    # hidden
    w1 = tf.Variable(
        tf.truncated_normal([NUM_FEATURES, hidden_units],
                            stddev=1.0 / np.sqrt(float(NUM_FEATURES))),
        name="weights"
    )
    b1 = tf.Variable(tf.zeros([hidden_units]),
                     name="bias")
    hidden = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(
        tf.truncated_normal([hidden_units, NUM_CLASSES],
                            stddev=1.0 / np.sqrt(float(hidden_units))),
        name="weights"
    )
    b2 = tf.Variable(tf.zeros([NUM_CLASSES]),
                     name="biase")
    
    y = tf.matmul(hidden, w2) + b2

    with tf.name_scope('corss_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_, logits=y)

    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    
    loss = tf.reduce_mean(cross_entropy + beta * regularization)
    return y, loss

def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return accuracy

