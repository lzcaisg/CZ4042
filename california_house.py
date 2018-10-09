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
hidden1 = 30

SEED = 10
np.random.seed(SEED)
DATA_DIR = 'cal_housing.data'

IMAGE_DIR = 'image'
CHECKPOINTS_DIR = 'q2_ckpts'


if not os.path.isdir(IMAGE_DIR):
	os.makedirs(IMAGE_DIR)

if not os.path.isdir(CHECKPOINTS_DIR):
	os.makedirs(CHECKPOINTS_DIR)


def get_train_test(test_radio=0.30):
	cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
	X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
	Y_data = (np.asmatrix(Y_data)).transpose()

	X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_radio, random_state=SEED)
	return X_train, X_test, y_train, y_test


def placehoder_inputs(num_features, num_labels):
    features_placeholder = tf.placeholder(tf.float32, [None, num_features])
    labels_placeholder = tf.placeholder(tf.float32, [None, num_labels])
    return features_placeholder, labels_placeholder


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=1.0 / np.sqrt(float(shape[0])))
	weights = tf.Variable(initial, dtype=tf.float32)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
	return weights


def bias_variable(shape):
	initial = tf.zeros(shape)
	return tf.Variable(initial, dtype=tf.float32)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
	logger.info("Add layer, shape: ({}, {})".format(input_dim, output_dim))
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = weight_variable([input_dim, output_dim])
		with tf.name_scope('biases'):
			biases = bias_variable([output_dim])
		u = tf.matmul(input_tensor, weights) + biases
		if act is None:
			return u
		else:	
			activations = act(u, name="activation")
			tf.summary.histogram('activations', activations)
			return activations

def hidden_layers(neurons, input_tensor):
	prev_layer = input_tensor
	for index, num_neurons in enumerate(neurons, 1):
		hidden = nn_layer(prev_layer, 
						  prev_layer.shape[1].value, 
						  num_neurons, 
						  "layer{}".format(index),
						  act=tf.nn.relu)
		prev_layer = hidden
	return hidden

def output_layer(input_tensor):
	y = nn_layer(input_tensor, input_tensor.shape[1].value, 1, "output", act=None)
	return y
	

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	logger.info('Train Test Split')
	X_train, X_test, y_train, y_test = get_train_test()
	logger.info('Train shape: {}, Test shape: {}, ratio: {}'.format(
		X_train.shape, X_test.shape, X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]))
	)

	x, y_ = placehoder_inputs(NUM_FEATURES, 1)
	h = hidden_layers(neurons=[30, 20, 20], input_tensor=x)
	y = output_layer(h)

	regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
	reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

	loss = tf.reduce_mean(tf.square(y_ - y)) + reg_term
	error = tf.reduce_mean(tf.square(y_ - y))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		train_err = []
		N = len(X_train)
		idx = np.arange(N)
		train_error_epoch = []

		writer = tf.summary.FileWriter("tmp/q2_models/5")
		writer.add_graph(sess.graph)

		'''
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
		'''


		'''
		TO-DO
		test_idx = np.arange(len(X_test))
		sample_idx = random.sample(test_idx, 50)
		X_sample, y_sample = X_test[test_idx], y_test[test_idx]
		y_pred = y.eval(feed_dict={x: X_sample})

		plt.figure(2)
		plot_targets = plt.plot(Y[:,0], Y[:,1], 'b^', label='targets')
		plot_pred = plt.plot(pred[:,0], pred[:,1], 'ro', label='predicted')
		plt.xlabel('$y_1$')
		plt.ylabel('$y_2$')
		plt.title('Sampled 50 target / predict')
		plt.legend()
		plt.savefig(os.join.path(IMAGE_DIR, "target_predicted"))

		'''

	'''
	kf = KFold(n_splits=5, shuffle=False)
	for train_index, val_index in kf.split(X):
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], Y[val_index]
	'''


