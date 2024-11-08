import numpy as np
from numpy import savetxt


def mse(expected, prediction):
    return np.mean((prediction - expected)**2)


def r2(expected, prediction):
    mean = np.mean(expected)
    res = np.sum((expected - prediction)**2)
    tot = np.sum((expected - mean)**2)
    return 1 - res/tot


def d_mse(expected, prediction):
    return 2/expected.size * (prediction - expected)


deriv = {
    mse: d_mse
}
