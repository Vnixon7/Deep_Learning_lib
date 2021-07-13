from numpy import ndarray as Tensor
import numpy as np
from typing import Dict, Callable
from typing import Iterator, NamedTuple
import sys


Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, batch_size: int=32, shuffle: bool=True) -> Iterator[Batch]:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starting = np.arange(0, len(inputs), self.batch_size)
        np.random.shuffle(starting)

        for start in starting:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)

class Optimizer:
    def __init__(self) -> None:
        pass
    def step(self, neuralnet) -> None:
        raise not NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate
        super().__init__()
    
    def step(self, neuralnet):
        for parameter, grad in neuralnet.parameters_and_grads():
            parameter -= self.learning_rate * grad

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
        
# total squared error
class TSE(Loss):
    # calculating the cost/loss
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2)
    # calculating gradient
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted - actual)



class Layer:
    def __init__(self, input_size: int, output_size: int, activation:str) -> None:
        self.parameters: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.parameters['weights'] = np.random.randn(input_size, output_size)
        self.parameters['bias'] = np.random.randn(output_size)
        #self.grads['weights'] = np.random.randn(input_size, output_size)
        #self.grads['bias'] = np.random.randn(output_size)
        self.activation = activation
        
        
        

    def activate(self, x):
        if self.activation == 'linear':
            return 0.001 * x
            
        if self.activation == 'tanh':
            return np.tanh(x)
            
        if self.activation == 'sigmoid':
            return 1/(1+np.exp(- x))

        if self.activation == 'softmax':
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

    def activate_dir(self, x):
        print(x)
        sys.exit()
        return self.activate(x) * (1 - self.activate(x))

    def forward_pass(self, inputs):
            self.layer_inputs = inputs
            print(self.layer_inputs)
            
            IWB =  inputs @ self.parameters['weights'] + self.parameters['bias']
            IWB = self.activate(IWB)
            return IWB

    
    def backward_pass(self, grad, inputs):
        self.layer_inputs = inputs
        self.grads["bias"] = np.sum(grad, axis=0)
        self.grads["weights"] = self.layer_inputs.T @ grad
        x = grad @ self.parameters["weights"].T
        print(x)
        x = self.activate_dir(x)
        return x

class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
            return inputs

    def backward_pass(self, grad, inputs):
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad, inputs)
        return grad

    def parameters_and_grads(self):
        for layer in self.layers:
            for name, parameter in layer.parameters.items():
                grad = layer.grads[name]
                yield parameter, grad


    def train(self,
          inputs: Tensor,
          targets: Tensor,
          epochs: int,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = TSE(),
          Optimizer: Optimizer = SGD(0.001)
          ) -> None:
        for epoch in range(epochs):
            x_loss = 0.0
            for batch in iterator(inputs, targets):
                predicted = self.forward_pass(batch.inputs)
                x_loss += loss.loss(predicted, batch.targets)
                grad = loss.grad(predicted, batch.targets)
                self.backward_pass(grad, batch.inputs)
                Optimizer.step(self)
            print(epoch,x_loss)


