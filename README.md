# pyneuron

Pyneuron is a small deep learning framework written in Python. It allows you to quickly build and test feedforward neural networks.

## Features
- Create layers of any size
- Dropout layer support
- Minibatch support
- Dynamic training data splitting
- Live training preview
- Periodic backup of model weights and biases
- Transfer learning support

## Available tools

<table>
  <tr>
    <th>Activation functions</th>
    <th>Optimizers</th>
    <th>Loss functions & scoring</th>
  </tr>
  <tr>
    <td>
      <ul>
        <li>ReLU</li>
        <li>Sigmoid</li>
        <li>Identity</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Stochastic Gradient Descent (SGD)</li>
        <li>Adam</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Mean Squared Error (MSE)</li>
        <li>R2 score</li>
      </ul>
    </td>
  </tr>
</table>

# Usage
## Preparing the environment
This framework depends mainly on Numpy and Pandas. You can automatically prepare a virtual environment for testing your models with:
```python
make venv
```

## Layer stack
You can construct a neural network by stacking layers on top of each other in an array. A neural network layer is a triplet:
```python
(<Layer output size>, <Activation fn>, <Dropout %>)
```

For example:
```python
regression = [
    (1, None, 0),
] 
```

This is the simplest example that represents linear regression. It takes any amount of parameters, doesn't pass them through any activation function and returns one output that's used for optimizing weights and biases.

A more complicated example would be:
```python
# count = input size
stack = [
    (count, relu, 0.2),
    (count, relu, 0.2),
    (count, relu, 0.3),
    (count, relu, 0.5),
    (count, relu, 0.5),
    (count, relu, 0.5),
    (count//2, relu, 0.5),
    (count//2, relu, 0.5),
    (count//2, relu, 0.3),
    (1, None, 0)
]
```

This 10-layer stack consists of:
- 6 ReLU layers with increasing dropout
- 3 ReLU layers with decreasing dropout and halved output size
- An output layer

## Constructing the neural network
The network is described by the `NeuralNetwork` structure:

```python
network = NeuralNetwork(input_size=count,
                        layer_data=stack,
                        loss_fn=mse,
                        optimizer=Adam,
                        minibatch_size=128)
```

It takes the following parameters:
- `input_size`: training input size
- `layer_data`: the layer stack
- `loss_fn`: loss function used for training (see Features)
- `optimizer`: model optimizer used for training (see Features)
- `minibatch_size`: minibatch size

## Model training
After constructing the model you can train it by calling the `train()` method. You need to supply the training data in a CSV format.

```python
network.train(training_data=data,
              output_vars=output,
              split=0.1,
              autosave=False)
```
It takes the following parameters:
- `training_data`: training data in a CSV format
- `output_vars`: values to optimize/train against
- `split`: ratio of training data allocated to the validation set
- `autosave`: enables periodic autosave to `.core` files

Training the model will start a live preview of its optimization progress:

<p align="center">
  <img src="/docs/resources/live-preview.png" width="75%">
</p>

After training the network you can save all the parameters by calling the `save_params()` method:

```python
network.save_params()
```

You can also load previously saved parameters to continue training on additional data for transfer learning:

```python
network.load_params("path/to/your/model.core")
```

Have fun!
