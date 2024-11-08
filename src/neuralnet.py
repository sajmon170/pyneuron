import numpy as np
import pandas as pd
import time
from multiprocessing.pool import ThreadPool
from collections import namedtuple

from layers import LayerData, LayerComposition
from loss import r2
from tools import unison_shuffle

SetData = namedtuple('SetData', ['inputs', 'outputs'])
            
    
class NeuralNetwork:
    def __init__(self, input_size, layer_data, loss_fn, optimizer, minibatch_size):
        self._loss_fn = loss_fn
        self._inputs = np.zeros([minibatch_size, input_size])
        self._evaluators = []
        self._minibatch_size = minibatch_size

        parameter_count = 0
        layer_input_size = input_size 
        for layer_tuple in layer_data:
            data = LayerData._make(layer_tuple)
            parameter_count += (layer_input_size + 1) * data.output_size
            layer_input_size = data.output_size

        self._parameters = np.random.rand(parameter_count).astype('float32')
        self._gradient = np.random.rand(parameter_count).astype('float32')
        self._optimizer = optimizer(parameter_count)
                 
        for i in range(minibatch_size):
            self._evaluators.append(LayerComposition(
                self._inputs[i].reshape(input_size, 1),
                layer_data,
                self._parameters,
                loss_fn))

                        
    def train_evaluator(self, evaluator, inputs, outputs):
        evaluator.forward_pass(inputs, training=True)
        evaluator.backprop(outputs)

        
    def evaluate(self, input_data):
        return self._evaluators[0].forward_pass(input_data)


    def __call__(self, input_data):
        return evaluate(input_data)
            

    def __epoch(self, training_set):
        sample_size = training_set.inputs.shape[0]

        index = 0
        for i in range(sample_size//self._minibatch_size):
            for j in range(self._minibatch_size):
                index += 1
                self.train_evaluator(self._evaluators[j],
                                     training_set.inputs[index],
                                     training_set.outputs[index])

            self._gradient[:] = np.zeros(self._gradient.size)
            
            for evaluator in self._evaluators:
                self._gradient += evaluator.get_gradient()

            self._gradient /= self._minibatch_size

            self._optimizer.optimize(self._gradient, self._parameters)
            unison_shuffle(training_set.inputs, training_set.outputs)

            
    def __validate(self, validation_set):
        evaluated = np.zeros(len(validation_set.inputs))
        
        index = 0
        for data in validation_set.inputs:
            evaluated[index] = self.evaluate(data)
            index += 1

        evaluated = evaluated.reshape(evaluated.size, 1)
        return self._loss_fn(validation_set.outputs, evaluated)


    def __get_score(self, validation_set):
        evaluated = np.zeros(len(validation_set.inputs))

        index = 0
        for data in validation_set.inputs:
            evaluated[index] = self.evaluate(data)
            index += 1

        evaluated = evaluated.reshape(evaluated.size, 1)

        print("Validation set sample:")
        print(validation_set.outputs[0:10])
        print("Corresponding predictions:")
        print(evaluated[0:10])

        return r2(validation_set.outputs, evaluated)


    def __prepare_data(self, dataframe, output_vars, split):
        def standardize(dataset):
            mean = np.mean(dataset, axis=0)
            std = np.std(dataset, axis=0)
            dataset[:] = np.nan_to_num((dataset - mean) / std)
            
        shuffled = dataframe.sample(frac=1)
        
        outputs_np = shuffled[output_vars].to_numpy(dtype=np.float32)
        inputs = shuffled.drop(columns=output_vars)
        inputs_np = inputs.to_numpy(dtype=np.float32) 

        training_out, validation_out \
            = np.split(outputs_np, [int(split*len(outputs_np))])
        
        training_in, validation_in \
            = np.split(inputs_np, [int(split*len(inputs_np))])

        standardize(training_in)
        standardize(training_out)
        standardize(validation_in)
        standardize(validation_out)
        
        training_data = SetData(training_in, training_out)
        validation_data = SetData(validation_in, validation_out)

        return training_data, validation_data
 
    
    def train(self, training_data, output_vars, split, autosave=False):
        training_set, validation_set = self.__prepare_data(training_data,
                                                           output_vars,
                                                           split)

        self.__epoch(training_set)
        prev_loss = self.__validate(validation_set)
        print("Epoch: 1")
        print(f"Loss: {prev_loss}")

        self.__epoch(training_set)
        current_loss = self.__validate(validation_set)
        print("Epoch: 2")
        print(f"Loss: {current_loss}")
        
        count = 3
        
        while True:
            self.__epoch(training_set)
            prev_loss = current_loss
            current_loss = self.__validate(validation_set)
            print(f"Epoch: {count}")
            print(f"Loss: {current_loss}")
            count += 1
            if count % 20 == 0:
                print(f"R2 rating: {self.__get_score(validation_set)}")
                if autosave:
                    self.save_params()


    def save_params(self, filename=None):
        if filename is None:
            filename = time.strftime("%Y%m%d-%H%M%S") + ".core"
        np.savetxt(filename, self._parameters)


    def load_params(self, filename):
        self._parameters[:] = np.loadtxt(filename)
