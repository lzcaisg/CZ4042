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


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	logger.info('Train Test Split')
	X_train, X_test, y_train, y_test = get_train_test()
	logger.info('Train shape: {}, Test shape: {}, ratio: {}'.format(
		X_train.shape, X_test.shape, X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]))
	)

	x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [None, 1])

	w1 = tf.Variable(
		tf.truncated_normal([NUM_FEATURES, hidden1], stddev=1.0 / np.sqrt(float(NUM_FEATURES))),
		dtype=tf.float32,
		name="weights1"
	)
	b1 = tf.Variable(
		tf.zeros([hidden1]),
		dtype=tf.float32,
		name="biases1"
	)
	h1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

	w2 = tf.Variable(
		tf.truncated_normal([hidden1, 1], stddev=1.0 / np.sqrt(float(hidden1))),
		dtype=tf.float32,
		name="weights2"
	)
	b2 = tf.Variable(
		tf.zeros([1]),
		dtype=tf.float32,
		name="biases2"
	)
	y = tf.add(tf.matmul(h1, w2), b2)

	regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
	loss = tf.reduce_mean(tf.square(y_ - y)) + beta * regularization
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

	kf = KFold(n_splits=5, shuffle=False)
	for train_index, val_index in kf.split(X):
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], Y[val_index]


