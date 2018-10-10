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
from PartB_q2 import train

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

def main():
    NUM_THREAD = mp.cpu_count() - 1
    with Pool(processes=NUM_THREAD) as pool:
        learning_rate = [1e-10 for _ in range(5)]
        num_neuron = [[20], [40], [60], [80], [100]]
        results = pool.starmap(train, zip(learning_rate, num_neuron))
    
    plt.figure()
    x = range(len(num_neuron))
    plt.plot(x, results)
    num_neurons_ticks = [str(_) for _ in num_neuron]
    plt.xticks(x, num_neurons_ticks)
    plt.xlabel('Number of hidden-layer neurons')
    plt.ylabel('Cross Validation Error')
    plt.title('Number of hidden-layers neurons vs Cross Validation Error')
    plt.savefig(os.path.join(IMAGE_DIR, "3. Number of hidden-layer neurons Cross Validation.png"))
    plt.show()


if __name__ == '__main__':
    main()