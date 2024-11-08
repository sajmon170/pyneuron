import numpy as np
from collections import namedtuple
from loss import deriv
from jvp import parameters_jvp, linear_jvp, jvp


LayerData = namedtuple('LayerData', ['output_size', 'function', 'dropout'])


class LayerComposition:    
    class Layer:
        def __init__(self, weights, bias, inputs, output_size, activation, dropout): 
            self.W = weights
            self.b = bias
            self.x = inputs
            self.f = activation
            self.linear = np.zeros(output_size, dtype=np.float32) \
                            .reshape(output_size, 1)
            self.out = np.zeros(output_size, dtype=np.float32) \
                         .reshape(output_size, 1)
            self.dropout = dropout


        def evaluate(self, training=False):
            np.matmul(self.W, self.x, out=self.linear)
            self.linear += self.b
            
            if training:
                mask = np.random.binomial(size=self.linear.shape, n=1, p=1-self.dropout)
                self.linear *= mask
                
            if self.f is not None:
                self.out[:] = self.f(self.linear)
            else:
                self.out[:] = self.linear

    
    def __init__(self, input_memory, layer_data, parameters, loss_fn):
        self._inputs = input_memory
        self._layers = []
        self._gradient = np.zeros(parameters.size)
        self._loss_fn = loss_fn

        index = 0
        inputs = self._inputs
        for layer_tuple in layer_data:
            data = LayerData._make(layer_tuple)

            weights_count = inputs.size * data.output_size
            weights = parameters[index:index + weights_count]
            weights = weights.reshape(data.output_size, inputs.size)
            index += weights_count
            
            biases = parameters[index:index + data.output_size]
            biases = biases.reshape(data.output_size, 1)
            biases.fill(0)
            index += data.output_size

            self._layers.append(self.Layer(weights, biases, inputs,
                                           data.output_size,
                                           data.function,
                                           data.dropout))

            inputs = self._layers[-1].out
    
    
    def forward_pass(self, input_data, training=False):
        self._inputs[:] = input_data.reshape(input_data.size, 1)

        for layer in self._layers:
            layer.evaluate(training=training)

        return self._layers[-1].out


    def backprop(self, expected):
        predicted = self._layers[-1].out
        delta = deriv[self._loss_fn](expected, predicted)

        index = self._gradient.size
        
        for layer in reversed(self._layers):
            if layer.f is not None:
                delta = jvp[layer.f](layer.linear, delta)

            biases = self._gradient[index - layer.b.size:index] \
                .reshape(layer.b.shape)
            index -= layer.b.size

            weights = self._gradient[index - layer.W.size:index] \
                .reshape(layer.W.shape)
            index -= layer.W.size
            
            parameters_jvp(layer.x, delta, weights, biases)
            delta = linear_jvp(layer.W, delta)


    def get_gradient(self):
        return self._gradient
