import numpy as cp
import scipy


def relu(x):
    return cp.maximum(0, x)


def sigmoid(x):
    return scipy.special.expit(x)
