import numpy as np
import scipy


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return scipy.special.expit(x)
