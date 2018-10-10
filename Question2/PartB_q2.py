import os
import logging
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from nn_models import placehoder_inputs, hidden_layers, output_layer
from PartB_q1 import get_data, split_train_test, get_batch_data

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
NUM_FEATURES = 8

batch_size = 32
learning_rate = 1e-7
epochs = 100
beta = 1e-3
num_neuron = [30]
no_folds = 5

SEED = 10
np.random.seed(SEED)

IMAGE_DIR = 'image'
if not os.path.isdir(IMAGE_DIR):
    logger.warn("image folder not exists, creating folder")
    os.makedirs(IMAGE_DIR)

def train(learning_rate, num_neuron):
    X_data, y_data = get_data()
    X_train, X_test, y_train, y_test = split_train_test(X_data, y_data, 0.3)

    # Create neural networks
    x, y_ = placehoder_inputs(NUM_FEATURES, 1)
    h = hidden_layers(neurons=num_neuron, input_tensor=x)
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

    kf = KFold(n_splits=no_folds)
    errors = []
    for train_index, test_index in kf.split(X_train):
        X, X_val = X_train[train_index], X_train[test_index]
        y, y_val = y_train[train_index], y_train[test_index]

        print(X.shape, y.shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # for i in tqdm(range(epochs)):
            for _ in range(epochs):
                data_gen = get_batch_data(X, y, batch_size)
                for X_batch, y_batch in data_gen:
                    sess.run(train_op, feed_dict={x: X_batch, y_: y_batch})
            
            err = error.eval(feed_dict={x: X_val, y_: y_val})
            errors.append(err)
    return sum(errors) / float(len(errors))

def main():
    NUM_THREAD = mp.cpu_count()
    with Pool(processes=NUM_THREAD) as pool:
        learning_rate = [0.5e-6, 1e-7, 0.5e-8, 1e-9, 1e-10]
        num_neuron = [[30] for _ in range(5)]
        results = pool.starmap(train, zip(learning_rate, num_neuron))
    
    plt.figure()
    x = range(len(learning_rate))
    plt.plot(x, results)
    lrticks = [str(_) for _ in learning_rate]
    plt.xticks(x, lrticks)
    plt.xlabel('Learning Rate')
    plt.ylabel('Cross Validation Error')
    plt.title('Learning Rate')
    plt.savefig(os.path.join(IMAGE_DIR, "2. Learning Rate Cross Validation.png"))
    plt.show()

if __name__ == "__main__":
    main()




