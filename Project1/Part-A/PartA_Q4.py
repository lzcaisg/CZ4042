# PartA-Q3: LzCai Oct/2018 U1622184H

# =====================
# ====== IMPORTS ======
# =====================

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import os
import time
from datetime import timedelta
import pickle
import logging

# ============================
# ====== SET PARAMETERS ======
# ============================

# Initialize Values
NUM_FEATURES = 36
NUM_CLASSES = 6
NUM_HIDDEN = 10

LEARNING_RATE = 0.01
EPOCHS = 2000
BATCH_SIZE = 4
SEED = 10
BETA = pow(10, -6)
np.random.seed(SEED)
DROP = True

TRAIN_FILE_NAME = 'sat_train.txt'
TEST_FILE_NAME = 'sat_test.txt'

file_name_label = "decay"
save_dir = "Z:\Github\CZ4042\save"

# =======================
# ====== LOAD DATA ======
# =======================

def scale(X, X_min, X_max): # scale data
    return (X - X_min)/(X_max-X_min)

def process_inputs_from_file(fileName): # Read in data
    inputs = np.loadtxt(fileName, delimiter=' ')
    X, _Y = inputs[:, :NUM_FEATURES], inputs[:, -1].astype(int)
    X = scale(X, np.min(X, axis=0), np.max(X, axis=0))
    _Y[_Y == 7] = 6 # Actually dont have, just in case have error data

    Y = np.zeros((_Y.shape[0], NUM_CLASSES))
    Y[np.arange(_Y.shape[0]), _Y - 1] = 1 #one hot matrix
    return X, Y

trainX, trainY = process_inputs_from_file(TRAIN_FILE_NAME)
testX, testY = process_inputs_from_file(TEST_FILE_NAME)
print ("Size of:")
print("- Training-set\t\t",len(trainX))
print("- Test-set\t\t",len(testX))

# ====================================
# ====== BUILD TENSORFLOW GRAPH ======
# ====================================

def init_weights(feature_no, neuron_no, name, logistic = True):
    # From eg.5.2
    n_in = feature_no
    n_out = neuron_no
    W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                            high=np.sqrt(6. / (n_in + n_out)),
                                            size=(n_in, n_out)))
    if logistic == True:
        W_values *= 4
    return(tf.Variable(W_values, dtype=tf.float32, name=name))

def init_bias(neuron_no, name):
    # From eg.5.2
    return(tf.Variable(np.zeros(neuron_no), dtype=tf.float32, name=name))

def setup_cross_entropy(labels, logits):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

def setup_correct_prediction(labels, logits):
    return tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32)

# Create Network
x = tf.placeholder(tf.float32, [None, NUM_FEATURES], name='x')
d = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='d')

# Saver
saver = tf.train.Saver()
if not os.path.exists(save_dir):
    print("Not Exist")
    os.makedirs(save_dir)

drop_str = "-Drop" if DROP else "-Not_Drop"
save_path = os.path.join(save_dir, str(EPOCHS)+ '-'+ str(BATCH_SIZE)+'-sigmoid')

# ============================
# ====== TENSORFLOW RUN ======
# ============================

def init_variables():
    session.run(tf.global_variables_initializer())

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def plot_graph(EPOCHS, BATCH_SIZE, acc_record, fileName, isTrain, error = False):
    if error:  
        acc_record = [1-tmp for tmp in acc_record]
        if isTrain:
            yLabel = 'Train error'
        else:
            yLabel = 'Test error'
    else:
        if isTrain:
            yLabel = 'Train accuracy'
        else:
            yLabel = 'Test accuracy'
    plt.figure(1)
    plt.plot(range(EPOCHS), acc_record)
    plt.xlabel(str(EPOCHS) + ' iterations')
    plt.ylabel(yLabel)
    plt.savefig(fileName)
    plt.show()

def validation_accuracy(testX, testY):
    output_2_, accuracy_ = session.run([y, accuracy], feed_dict={x: testX, d: testY})
    print(output_2_, '\n',accuracy_)

train_acc_backup = []
test_acc_backup = []
time_usage_backup = []
total_time_backup = []

decay_list = [0, pow(10, -3), pow(10, -6), pow(10, -9), pow(10, -12)]
graph_list = [None for i in range(len(decay_list))]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler('./'+file_name_label+'.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.log(level=logging.INFO,msg= "=============== START ===============")

for i in range(len(decay_list)):
    graph_list[i] = tf.Graph()
    NUM_HIDDEN = decay_list[i]
    with graph_list[i].as_default():
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES], name='x')
        d = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='d')
        
        with tf.variable_scope("Hidden_layer"):
            W = init_weights(NUM_FEATURES, NUM_HIDDEN, name="Weight_1")
            b = init_bias(NUM_HIDDEN, name="Bias_1")
            z = tf.matmul(x, W) + b #syn_input_1
            h = tf.nn.sigmoid(z) #out_1

        with tf.variable_scope("Output_layer"):
            V = init_weights(NUM_HIDDEN, NUM_CLASSES, name="Weight_2")
            c = init_bias(NUM_CLASSES, name="Bias_2" )
            u = tf.matmul(h, V) + c #syn_out_2
            y = tf.nn.sigmoid(u) #out_2  # Consider to change to sigmoid
            
        cross_entropy = setup_cross_entropy(labels=d, logits=y)
        regularization = tf.nn.l2_loss(V) + tf.nn.l2_loss(W) 
        J = tf.reduce_mean(cross_entropy + BETA * regularization)
        correct_prediction = setup_correct_prediction(labels=d, logits=y)
        accuracy = tf.reduce_mean(correct_prediction)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(J)
        
        saver = tf.train.Saver()
        session = tf.Session()
        init_variables()
        train_acc = []
        save_path = os.path.join(save_dir, str(EPOCHS)+ '-'+ str(NUM_HIDDEN)+'-hidden')

        with session.as_default():
            # Ensure we update the global variables rather than local copies.
            logger.log(level=logging.INFO,msg= "Hidden-Neurons: "+str(NUM_HIDDEN))
            # Start-time used for printing time-usage below.
            start_time = time.time()
            train_acc_record = []
            test_acc_record = []
            epoch_time_record = []

            best_test_acc = 0.0
            last_improvement = 0
            improved_str = ""
            test_count = 0

            mul = int(len(trainX)/BATCH_SIZE)
            for i in range(EPOCHS):

                epoch_start_time = time.time()
                for j in range(mul):
                    x_batch, d_batch = next_batch(BATCH_SIZE, trainX, trainY)
                    feed_dict_train = {x: x_batch, d: d_batch}
                    session.run(train_op, feed_dict=feed_dict_train)

                train_acc_record.append(accuracy.eval(feed_dict=feed_dict_train))
                epoch_end_time = time.time()
                epoch_time_diff = epoch_end_time-epoch_start_time
                epoch_time_record.append(epoch_time_diff)

                if i % 100 == 0 or i == (EPOCHS - 1):
                    test_count += 1
                    test_accuracy = session.run(accuracy, feed_dict={x: testX, d: testY})
                    test_acc_record.append(test_accuracy)
                    if DROP:
                        if test_accuracy > best_test_acc:
                            best_test_acc = test_accuracy
                            last_improvement = i
                            saver.save(sess=session, save_path=save_path)
                            improved_str = "*"
                        else:
                            improved_str = ''
                    else:
                        saver.save(sess=session, save_path=save_path)

                    logger.log(level=logging.INFO, msg='iter %d: Train accuracy %g'%(i, train_acc_record[i])+' Test accuracy: '+str(test_accuracy)+str(improved_str))



        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        logger.log(level=logging.INFO, msg="Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        
        train_graphName = "PartA-Train"+str(EPOCHS)+'-'+str(BATCH_SIZE)+".png"
        test_graphName = "PartA-Test"+str(EPOCHS)+'-'+str(BATCH_SIZE)+".png"
        plot_graph(EPOCHS, BATCH_SIZE, train_acc_record,train_graphName, isTrain = True, error=True)
        plot_graph(test_count, BATCH_SIZE, test_acc_record, test_graphName, isTrain = False)

        train_acc_backup.append(train_acc_record)
        test_acc_backup.append(test_acc_record)
        time_usage_backup.append(epoch_time_record)
        total_time_backup.append(time_dif)

        #=========== Save all the data for EACH TRAINING Has Done ============#
        fileNameTail = str(EPOCHS)+'-'+str(NUM_HIDDEN)+file_name_label+".out"

        train_acc_filename = "PartA-Train_Acc-"+fileNameTail
        with open(train_acc_filename, 'wb') as fp:
            pickle.dump(train_acc_backup, fp)

        test_acc_filename = "PartA-Test_Acc-"+fileNameTail
        with open(train_acc_filename, 'wb') as fp:
            pickle.dump(test_acc_backup, fp)

        time_usage_filename = "PartA-Time_Usage-"+fileNameTail
        with open(time_usage_filename, 'wb') as fp:
            pickle.dump(time_usage_backup, fp)

        time_usage_filename = "PartA-Time_Usage-"+fileNameTail
        with open(time_usage_filename, 'wb') as fp:
            pickle.dump(time_usage_backup, fp)  

logger.log(level=logging.INFO,msg= "=============== END ===============")
        