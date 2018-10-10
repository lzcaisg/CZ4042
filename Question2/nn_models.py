import os
import numpy as np
import random
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def placehoder_inputs(num_features, num_labels):
	'''Create placeholders for the input
	
	Args:
		num_features (int): number of the input features 
		num_labels (int): number of the out labels
	
	Returns:
		tf.placeholder: placeholders 
	'''
	
	features_placeholder = tf.placeholder(tf.float32, [None, num_features])
	labels_placeholder = tf.placeholder(tf.float32, [None, num_labels])
	return features_placeholder, labels_placeholder


def weight_variable(shape):
	'''Create a weight variable with the shape
	
	Args:
		shape (list): the shape of the weight
	
	Returns:
		tf.Variable: a weight variable
	'''

	initial = tf.truncated_normal(shape, stddev=1.0 / np.sqrt(float(shape[0])))
	weights = tf.Variable(initial, dtype=tf.float32)
	tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
	return weights


def bias_variable(shape):
	'''Create a bias variable
	
	Args:
		shape (list): shape of the bias variable
	
	Returns:
		tf.Variable: a bias variable
	'''

	initial = tf.zeros(shape)
	return tf.Variable(initial, dtype=tf.float32)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, 
			 keep_prob=None, act=tf.nn.relu):
	'''Create a fully conncected layer for the model
	
	Args:
		input_tensor (tf.Variable): input tensor
		input_dim (int): input dimensions of the layer
		output_dim (int): output dimensions of the layer
		layer_name (str): name
		act (function): Defaults to tf.nn.relu. activation function
	
	Returns:
		tf.Tensor: output tensor
	'''

	logger.info("Add layer, shape: ({}, {})".format(input_dim, output_dim))
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = weight_variable([input_dim, output_dim])
		with tf.name_scope('biases'):
			biases = bias_variable([output_dim])
		u = tf.matmul(input_tensor, weights) + biases
		if act is None:
			activations = u
		else:	
			activations = act(u, name="activation")

		out = activations
		if keep_prob is not None:
			out = tf.nn.dropout(out, keep_prob, name="dropout")
		return out

def hidden_layers(neurons, input_tensor, keep_prob=None):
	'''Create hidden layers for the model. Connect all the hidden layers created. 
	
	Args:
		neurons (list): a list of each layer's hidden neuron numbers
		input_tensor (tf.Tensor): input tensor
	
	Returns:
		tf.Tensor: the last hidden layers
	'''

	prev_layer = input_tensor
	for index, num_neurons in enumerate(neurons, 1):
		hidden = nn_layer(
			prev_layer, 
			prev_layer.shape[1].value, 
			num_neurons, 
			"layer{}".format(index),
			keep_prob=None,
			act=tf.nn.relu
		)
		prev_layer = hidden
	return hidden

def output_layer(input_tensor, output_dim=1):
	'''Create ouput layers of the model
	
	Args:
		input_tensor (tf.Tensor): 
	
	Returns:
		[tf.Tensor]: model output
	'''

	y = nn_layer(input_tensor, input_tensor.shape[1].value, output_dim, "output", act=None)
	return y

def training(loss, learning_rate):
	# Create the gradient descent optimizer with the learning rate
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	return train_op


def evaluation(output, target):
	error = tf.reduce_mean(tf.square(target - output))
	return error
