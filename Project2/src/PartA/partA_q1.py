import os
import pickle
import numpy as np
import tensorflow as tf

NUM_CLASSES = 10
IMG_SIZE = 10
NUM_CHANNELS = 3
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

BATCH_SIZE = 128
LEARNING_RATE = 0.001

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return data, labels

def process_inputs(data, labels):
    '''Process inputs for the neural network

    Returns:
        image: 4D tensor with shape [N, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS] (NHWC)
        labels: 1D tensor with shape [N, NUM_CLASSES]
    '''

    # one-hot encoding
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1
    images = tf.reshape(data, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    return images, labels_

def weight_variable(shape, stddev, name="weights"):
    initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bias"):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name=name)

def inference(images):
    with tf.variable_scope("conv1") as scope:
        w = weight_variable(shape=[9, 9, NUM_CHANNELS, 50],
                            stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9),
                            name="weights1")
        b = bias_variable(shape=[50], name="bias1")
        conv = tf.nn.conv2d(images, w, [1, 1, 1, 1], padding='VALID')
        pre_activation = tf.nn.bias_add(conv, b)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                           padding='VALID', name='pool1')
        
    with tf.variable_scope("conv2") as scope:
        w = weight_variable(shape=[5, 5, NUM_CHANNELS, 60],
                            stddev=1.0 / np.sqrt(NUM_CHANNELS * 5 * 5),
                            name="weights2")
        b = bias_variable(shape=[60], name="bias2")
        conv = tf.nn.conv2d(pool1, w, [1, 1, 1, 1], padding="VALID")
        pre_activation = tf.nn.bias_add(conv, b)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding="VALID", name="pool2")

    with tf.variable_scope("fc1") as scope:
        pool2_shape = pool2.get_shape().as_list()
        dim = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
        reshape = tf.reshape(pool2, [-1, dim])
        w = weight_variable(shape=[dim, 300], stddev=1.0 / np.sqrt(dim), name="weights3")
        b = bias_variable(shape=[300])
        fc1 = tf.nn.relu(tf.matmul(reshape, w) + b, name=scope.name)

    with tf.variable_scope('softmax_linear'):
        w = weight_variable(shape=[300, NUM_CLASSES], stddev=1.0 / np.sqrt(300), name="weights4")
        b = bias_variable(shape=[NUM_CLASSES], name="bias4")
        softmax_linear = tf.add(tf.matmul(fc1, w), b, name=scope.name)
    
    return softmax_linear

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy)
    return loss

def accuracy(preidctions, label):




def train():
    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)

    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)
    
    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
    
    # Convolutional Neural Network
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    logits = inference(x)
    loss = loss(logits=logits, labels=y_)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
