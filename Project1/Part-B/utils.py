import numpy as np
from sklearn.model_selection import train_test_split

SEED = 10


def get_batch_data(X_train, y_train, batch_size):
    idx = np.arange(len(X_train))
    N = len(X_train)
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]

    for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
        yield X_train[start: end], y_train[start: end]


def get_data(data_dir):
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
    X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
    Y_data = (np.asmatrix(Y_data)).transpose()
    return X_data, Y_data


def split_train_test(X_data, Y_data, test_ratio=0.30):
    '''Split the validating and testing datasets
        test_radio (float, optional): Defaults to 0.30. ratio of test datasets
    '''
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, 
    	                                                test_size=test_ratio, random_state=SEED)
    return X_train, X_test, y_train, y_test
