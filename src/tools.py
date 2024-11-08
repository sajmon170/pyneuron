import numpy as np


def count_nonmatching(data, matches):
    return len(list(filter(lambda x: x not in matches, data)))


def unison_shuffle(a, b):
    p = np.random.permutation(len(a))
    a = a[p]
    b = b[p]
