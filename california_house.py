import numpy as np
import tensorflow as tf
import pylab as plt

from sklearn.model_selection import train_test_split

NUM_FEATURES = 8

learning_rate = 0.01
epochs = 500
batch_size = 32
num_neuron = 30
seed = 10
np.random.seed(seed)

def prepare_data():
	#read and divide data into test and train sets 
	cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
	X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
	Y_data = (np.asmatrix(Y_data)).transpose()

	X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=seed)
	return X_train, X_test, y_train, y_test

def main(hidden1=num_neuron):
# Create the model
	x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [None, 1])

	w1 = tf.Variable(
		tf.truncated_normal([NUM_FEATURES, hidden1], stddev=1.0 / np.sqrt(float(NUM_FEATURES))),
		dtype=tf.float32,
		name="weights"
	)
	b1 = tf.Variable(
		tf.zeros([hidden1]),
		name="biases"
	)
	h1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

	w2 = tf.Variable(
		tf.truncated_normal([hidden1, 1], stddev=1.0 / np.sqrt(float(hidden1))),
		dtype=tf.float32,
		name="weights"
	)
	b2 = tf.Variable(
		tf.zeros([1]),
		name="biases"
	)
	y = tf.add(tf.matmul(h1, w2), b2)	
	regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
	loss = tf.reduce_mean(tf.square(y_ - y)) + regularization
	error = tf.reduce_mean(tf.square(y_ - y))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)


	X_train, X_test, y_train, y_test = prepare_data()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		train_err = []
		N = len(X_train)
		idx = np.arange(N)

		for i in range(epochs):
			np.random.shuffle(idx)
			X_train, y_train = X_train[idx], y_train[idx]
			print(i)
			for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
				train_op.run(feed_dict={x: X_train[start: end], y_: y_train[start: end]})
				err = error.eval(feed_dict={x: X_test, y_: y_test})
				train_err.append(err)
	
			if i % 100 == 0:
				print('iter %d: test error %g'%(i, train_err[i]))

	# plot learning curves
	plt.figure(1)
	plt.plot(range(epochs), train_err)
	plt.xlabel(str(epochs) + ' iterations')
	plt.ylabel('Train Error')
	plt.show()

main()
