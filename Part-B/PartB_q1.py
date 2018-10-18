import matplotlib as mpl
mpl.use('Agg')

import os
import logging
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from nn_models import placehoder_inputs, hidden_layers, output_layer
from utils import get_batch_data, get_data
import nn_models

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Hyperparameters
NUM_FEATURES = 8
BATCH_SIZE = 256
LEARNING_RATE = 1e-7
EPOCHS = 2500
BETA = 1e-3

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seed
SEED = 10
np.random.seed(SEED)

# Data directory
DATA_DIR = 'cal_housing.data'

# Image directory
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

    y_test /= 1000
    y_train /= 1000


    # Create the neural network
    x, y_ = placehoder_inputs(NUM_FEATURES, 1)
    # Create hidden layers
    # Pass in parameter neurons=[30]: one hidden layer, contains 30 hiden neurons 
    #h = hidden_layers(neurons=[30], input_tensor=x)
    #y = output_layer(h)

    w1 = nn_models.weight_variable([NUM_FEATURES, 30])
    b1 = nn_models.bias_variable([30])
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = nn_models.weight_variable([30, 1])
    b2 = nn_models.bias_variable([1])
    y = tf.matmul(h1, w2) + b2

    # L2 Regularization
    regularizer = tf.contrib.layers.l2_regularizer(scale=BETA)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

    # Define loss
    loss = tf.reduce_mean(tf.square(y_ - y)) + reg_term

    # Define error
    error = tf.reduce_mean(tf.square(y_ - y))

    # Create the gradient descent optimizer with the learning rate
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

    train_op = optimizer.minimize(loss)

    # Ops to save and restore models
    saver = tf.train.Saver()

    # Train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        train_error_epoch = []


        # Store model for tensorboard
        writer = tf.summary.FileWriter(os.path.join("tmp", "part2_models", "q1"))
        writer.add_graph(sess.graph)

        for i in tqdm(range(EPOCHS)):
        	# Get a banch of data with batch_size = 32
            data_gen = get_batch_data(X_train, y_train, BATCH_SIZE)
            # initialize train_error for each epoch
            train_error = 0
            for X_batch, y_batch in data_gen:
                train_op.run(feed_dict={x: X_batch, y_: y_batch})
                # Evaluation train error
                train_error += error.eval(feed_dict={x: X_batch, y_: y_batch})
            
            # append train error for each epoch
            train_error_epoch.append(train_error)

            if i % 1000 == 0:
                logger.info('epoch:{}, training error:{}'.format(i, train_error_epoch[i]))
                print(y.eval(feed_dict={x: X_test[:10]}))
                print(y_test[:10])

        # Save model checkpoints
        save_path = saver.save(sess, os.path.join("tmp", "part2_ckpts", "q1.ckpt"))

    # Restore the model from checkpoints
    # with tf.Session() as sess:
    #    saver.restore(sess, os.path.join("tmp", "part2_ckpts", "q1.ckpt"))
    #    logger.info("Model restored")

        # Randomly get 50 test samples
        test_idx = np.arange(len(X_test))
        sample_idx = np.random.choice(test_idx, 50)
        X_sample, y_sample = X_test[sample_idx], y_test[sample_idx]

        y_pred = y.eval(feed_dict={x: X_sample})

        prediction = y_pred.flatten()
        target = np.asarray(y_sample).reshape(-1)

        print(list(zip(prediction, target)))

        # Plot training errors
        logger.info("Plotting training errors")
        plt.figure(1)
        plt.plot(range(EPOCHS), train_error_epoch)
        plt.xlabel(str(EPOCHS) + ' iterations')
        plt.ylabel('Train Error')
        plt.savefig(os.path.join(IMAGE_DIR, "PartB_q1a.png"))

        # Plot target vs prediction
        logger.info("Plotting target vs prediction")
        plt.figure(2)
        plt.scatter(target, prediction)
        # zero error line
        max_test = max(max(target), max(prediction))
        plt.plot([0, max_test], [0, max_test], color='r')
        plt.xlabel('Prediction')
        plt.ylabel('Target')
        plt.title('Prediction Target Plot')
        plt.savefig(os.path.join(IMAGE_DIR, "PartB_q1b.png"))


