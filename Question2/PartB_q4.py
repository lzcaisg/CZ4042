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
# Optimal number of hidden layer neurons
num_neuron = [20]
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
        beta = [0.0, 1e-3, 1e-6, 1e-9, 1e-12]
        learning_rate = [1e-10 for _ in range(len(beta))]
        num_neuron = [[20] for _ in range(len(beta))]
        results = pool.starmap(train, zip(learning_rate, num_neuron, beta))

    plt.figure()
    x = range(len(beta))
    plt.plot(x, results)
    beta_ticks = [str(_) for _ in beta]
    plt.xticks(x, beta_ticks)
    plt.xlabel('weight decay')
    plt.ylabel('Cross Validation Error')
    plt.title('Weight Decay vs Cross Validation Error')
    plt.savefig(os.path.join(IMAGE_DIR, "4.Weight Decay Cross Validation.png"))
    plt.show()

if __name__ == '__main__':
    main()