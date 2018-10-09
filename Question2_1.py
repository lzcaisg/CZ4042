from nn_models import placehoder_inputs, hidden_layers, output_layer
import os
import numpy as np
import random
import tensorflow as tf
import pylab as plt
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split, KFold

NUM_FEATURES = 8

learning_rate = 1e-7
epochs = 500
batch_size = 32
beta = 1e-3
num_neuron = 30
no_folds = 5

SEED = 10
np.random.seed(SEED)

IMAGE_DIR = 'image'
CHECKPOINTS_DIR = 'q2_ckpts'
if not os.path.isdir(IMAGE_DIR):
	os.makedirs(IMAGE_DIR)

if not os.path.isdir(CHECKPOINTS_DIR):
	os.makedirs(CHECKPOINTS_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_train_test(test_radio=0.30):
	cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
	X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
	Y_data = (np.asmatrix(Y_data)).transpose()

	X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_radio, random_state=SEED)
	return X_train, X_test, y_train, y_test

# Prepare validaing and testing datasets
logger.info('Train Test Split')
X_train, X_test, y_train, y_test = get_train_test()
logger.info('Train shape: {}, Test shape: {}, ratio: {}'.format(
    X_train.shape, X_test.shape, X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]))
)

# Create the model
x, y_ = placehoder_inputs(NUM_FEATURES, 1)
h = hidden_layers(neurons=[30,20,50], input_tensor=x)
y = output_layer(h)

# L2 Regularization
regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

# Loss
loss = tf.reduce_mean(tf.square(y_ - y)) + reg_term

# Error
error = tf.reduce_mean(tf.square(y_ - y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
saver = tf.train.Saver()

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_err = []
    N = len(X_train)
    idx = np.arange(N)
    train_error_epoch = []

    writer = tf.summary.FileWriter("tmp/q2_models/5")
    writer.add_graph(sess.graph)


    for i in tqdm(range(epochs)):
        np.random.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]
        train_error = 0
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
            train_op.run(feed_dict={x: X_train[start: end], y_: y_train[start: end]})
            train_error += error.eval(feed_dict={x: X_train[start: end], y_: y_train[start: end]})
            
        train_error_epoch.append(train_error)
        #if i % 100 == 0:
        # 	print('iter %d: test error %g'%(i, train_error_epoch[i]))

    save_path = saver.save(sess, os.path.join(CHECKPOINTS_DIR, "model.ckpt"))
    logger.info("Model saved")

    plt.figure(1)
    plt.plot(range(epochs), train_error_epoch)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Train Error')
    plt.show()


    kf = KFold(n_splits=5, shuffle=False)
    for train_index, val_index in kf.split(X):
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], Y[val_index]