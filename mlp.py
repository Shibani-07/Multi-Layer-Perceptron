import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple

def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """

    np.random.shuffle(np.arange(train_x.shape[0]))
    for key in range(0, train_x.shape[0], batch_size):
        limit = min(key + batch_size, train_x.shape[0])
        batch = np.arange(train_x.shape[0])[key:limit]
        yield train_x[batch], train_y[batch]

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)*(1-self.forward(x))


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (1-self.forward(x)**2)


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0,x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)*(1-self.forward(x))


class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class Softplus(ActivationFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x))


class Mish(ActivationFunction):
    def forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    def derivative(self, x):
        omega = 4 * (x + 1) + np.exp(2 * x) + np.exp(x) * (4 * x + 6)
        delta = 2 * np.exp(x) + np.exp(2 * x) + 2
        return np.exp(x) * omega / (delta ** 2)


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true



class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.mean(y_true * np.log(y_pred + 1e-9))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate=0.0):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        
        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        self.z = None
        

        # Initialize weights and biaes

        l = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-l, l, (fan_in, fan_out))  # Glorot Uniform Initialization
        self.b = np.zeros((1, fan_out))

    def forward(self, h: np.ndarray, training=True):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        self.z = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(self.z)

        if training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.activations.shape) / (1 - self.dropout_rate)
            self.activations *= self.dropout_mask
        

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        dL_dz = delta * self.activation_function.derivative(self.z)
        if self.dropout_rate > 0.0:
            dL_dz *= self.dropout_mask
        dL_dW = np.dot(h.T, dL_dz)
        dL_db = np.sum(dL_dz, axis=0, keepdims=True)
        self.delta = np.dot(dL_dz, self.W.T)
        return dL_dW, dL_db

class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer.forward(x,training=training)
        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []
        delta = loss_grad

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            a = input_data if i == 0 else self.layers[i - 1].activations
            dl_dw, dl_db = layer.backward(a, delta)
            dl_dw_all.insert(0, dl_dw)
            dl_db_all.insert(0, dl_db)
            delta = layer.delta

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, rmsprop=False, beta=0.9, epsilon=1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = []
        validation_losses = []
        velocity_w = [np.zeros_like(layer.W) for layer in self.layers]
        velocity_b = [np.zeros_like(layer.b) for layer in self.layers]

        num_batches = int(np.ceil(train_x.shape[0] / batch_size))  

        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in batch_generator(train_x, train_y, batch_size):
                y_pred = self.forward(x, training=True)
                epoch_loss += loss_func.loss(y, y_pred)

                loss_grad = loss_func.derivative(y, y_pred)
                dl_dw_all, dl_db_all = self.backward(loss_grad, x)

                for i, layer in enumerate(self.layers):
                  if rmsprop:
                      velocity_w[i] = beta * velocity_w[i] + (1 - beta) * (dl_dw_all[i] ** 2)
                      velocity_b[i] = beta * velocity_b[i] + (1 - beta) * (dl_db_all[i] ** 2)

                      layer.W -= learning_rate * dl_dw_all[i] / (np.sqrt(velocity_w[i]) + epsilon)
                      layer.b -= learning_rate * dl_db_all[i] / (np.sqrt(velocity_b[i]) + epsilon)
                  else:
                      layer.W -= learning_rate * dl_dw_all[i]
                      layer.b -= learning_rate * dl_db_all[i]

            training_losses.append(epoch_loss / num_batches)
            val_loss = loss_func.loss(val_y, self.forward(val_x, training=True))
            validation_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {training_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}")


        return training_losses, validation_losses
