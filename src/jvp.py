import numpy as np
from activation import sigmoid, relu


def parameters_jvp(inputs, delta, out_dw, out_db):
    np.matmul(delta, inputs.transpose(), out=out_dw)
    out_db[:] = delta
    

def linear_jvp(weights, delta):
    result = np.matmul(weights.transpose(), delta)
    return result


def relu_jvp(inputs, delta):
    return (inputs > 0) * delta


def sigmoid_jvp(inputs, delta):
    return (1 - sigmoid(inputs)) * sigmoid(inputs) * delta


jvp = {
    relu: relu_jvp,
    sigmoid: sigmoid_jvp
}
