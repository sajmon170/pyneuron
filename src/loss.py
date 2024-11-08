import numpy as cp
from numpy import savetxt


def mse(expected, prediction):
    return cp.mean((prediction - expected)**2)


def r2(expected, prediction):
    mean = cp.mean(expected)
    res = cp.sum((expected - prediction)**2)
    tot = cp.sum((expected - mean)**2)
    return 1 - res/tot


def d_mse(expected, prediction):
    return 2/expected.size * (prediction - expected)


deriv = {
    mse: d_mse
}
