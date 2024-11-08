from neuralnet import NeuralNetwork
from activation import sigmoid, relu
from loss import mse
from optimizer import GradientDescent, Adam
from tools import count_nonmatching
import pandas as pd
import numpy as np


def main():
    output = ['price']
    
    """
    original_sample= pd.read_csv('dataset/training/audi.csv')
    trait_count = len(original_sample.columns)
    
    data = pd.read_csv('dataset/training/training.csv')
    count = count_nonmatching(data.columns, output)
    """
    
    data = pd.read_csv('dataset/training/audi_transformed.csv')
    count = count_nonmatching(data.columns, output)
    trait_count = count
    
    regression = [
        (1, None, 0),
    ] 
    
    structure1 = [
        (count, relu, 0.2),
        (count, relu, 0.2),
        (count, relu, 0.3),
        (trait_count, relu, 0.5),
        (trait_count, relu, 0.5),
        (trait_count, relu, 0.5),
        (trait_count//2, relu, 0.5),
        (trait_count//2, relu, 0.5),
        (trait_count//2, relu, 0.3),
        (1, None, 0)
    ]

    structure2 = [
        (count, relu, 0.2),
        (count, relu, 0.5), 
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5), 
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5),
        (count, relu, 0.5),
        (1, None, 0)
    ]

    structure3 = [
        (count, relu, 0.2),
        (trait_count, relu, 0.5),
        (trait_count//2, relu, 0.5),
        (trait_count//2, relu, 0.5),
        (trait_count//4, relu, 0.5),
        (trait_count//4, relu, 0.5),
        (1, None, 0) 
    ]

    testing_random = [
        (count, sigmoid, 0.1),
        (count, relu, 0.2),
        (count, relu, 0.2),
        (count, relu, 0.2),
        (count, relu, 0.2),
        (1, None, 0)
    ]

    testing = [
        (count, sigmoid, 0),
        (count, relu, 0),
        (count, relu, 0),
        (1, None, 0)
    ]
    
    network = NeuralNetwork(input_size=count,
                            layer_data=testing,
                            loss_fn=mse,
                            optimizer=Adam,
                            minibatch_size=128)

    network.train(training_data=data,
                  output_vars=output,
                  split=0.1,
                  autosave=False)

    network.save_params()


if __name__ == "__main__":
    np.random.seed(1234)
    main()
